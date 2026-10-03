"""LiDAR front-end: keyframe selection, per-keyframe point extraction, per-keyframe Manhattan/floor measurements."""
import logging
from dataclasses import dataclass
import numpy as np
import cv2

from ..geom import voxel_downsample, rot_y, robust_mode_1d
from ..io.record3d import backproject

log = logging.getLogger("roomscan")


@dataclass
class Keyframe:
    cap_idx: int
    t: float
    R0: np.ndarray            # original camera->world rotation (3x3)
    p0: np.ndarray            # original camera position (3,)
    pts_cam: np.ndarray       # (M,3) float32 camera-frame points (voxel-thinned)
    edge_cam: np.ndarray      # (E,3) float32 near-side points of vertical depth discontinuities (jamb candidates)
    psi0: float               # original heading (yaw about world Y) from camera right axis
    man_theta: float          # local dominant horizontal-normal angle (rad, mod 90deg), nan if unreliable
    man_R: float              # concentration in [0,1]
    floor_y: float            # measured floor height in world (original pose), nan if unseen
    floor_n: int              # pixels supporting floor_y


def heading_of(R):
    r = R[:, 0]
    return float(np.arctan2(-r[2], r[0]))


def select_keyframes(cap, cfg):
    idx = np.where(cap.has_depth)[0]
    if len(idx) == 0:
        raise RuntimeError("capture has no depth frames")
    keep = [idx[0]]
    for i in idx[1:]:
        j = keep[-1]
        dt = np.linalg.norm(cap.T_c2w[i, :3, 3] - cap.T_c2w[j, :3, 3])
        Rrel = cap.T_c2w[j, :3, :3].T @ cap.T_c2w[i, :3, :3]
        ang = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        if dt >= cfg.kf_min_trans_m or ang >= cfg.kf_min_rot_deg:
            keep.append(i)
    keep = np.array(keep)
    if len(keep) > cfg.max_keyframes:
        keep = keep[np.linspace(0, len(keep) - 1, cfg.max_keyframes).astype(int)]
    return keep


def _normals_from_depth(d, K):
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    ds = cv2.GaussianBlur(d, (3, 3), 0)
    P = np.stack([(u - K[0, 2]) / K[0, 0] * ds, (v - K[1, 2]) / K[1, 1] * ds, ds], -1)
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    # orient normals toward the camera
    flip = (n * P[1:-1, 1:-1]).sum(-1) > 0
    n[flip] *= -1
    return n, nn[..., 0]


def edge_points(d, conf, K, cfg, min_jump_m=0.30, min_rows=6):
    """Near-side pixels of VERTICAL depth discontinuities (door/window jambs, wall ends, furniture sides).
    Taken before flying-pixel filtering; the near side of a step edge lies on the true edge to within half a pixel."""
    H, W = d.shape
    valid = (d > cfg.depth_min_m) & (d < cfg.depth_max_m)
    if conf is not None:
        valid &= conf >= cfg.min_confidence
    a, b = d[:, :-1], d[:, 1:]
    va, vb = valid[:, :-1], valid[:, 1:]
    jump = va & vb & (np.abs(a - b) > np.maximum(min_jump_m, 0.10 * np.minimum(a, b)))
    near_is_a = a < b
    m = np.zeros((H, W), bool)
    rr, cc = np.nonzero(jump)
    cn = np.where(near_is_a[rr, cc], cc, cc + 1)
    m[rr, cn] = True
    # keep only edges that persist vertically (>= min_rows consecutive rows): jambs are tall straight lines
    k = np.ones((min_rows, 1), np.uint8)
    keep = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    v, u = np.nonzero(keep)
    if len(v) == 0:
        return np.zeros((0, 3), np.float32)
    z = d[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]; y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1).astype(np.float32)


def process_keyframe(cap, i, cfg, floor_guess=None):
    d = cap.depth_m(i)
    if d is None:
        return None
    conf = cap.conf(i)
    K = cap.K_depth(i)
    pc, (uu, vv), m = backproject(d, K, conf, cfg.depth_min_m, cfg.depth_max_m, cfg.min_confidence, cfg.edge_rel_thresh)
    if len(pc) < 200:
        return None
    R0, p0 = cap.T_c2w[i, :3, :3], cap.T_c2w[i, :3, 3]
    # ---- Manhattan measurement from surface normals (world horizontal normals) ----
    n, nn = _normals_from_depth(d, K)
    mm = m[1:-1, 1:-1] & (nn > 1e-7)
    nw = n[mm] @ R0.T
    h = nw[np.abs(nw[:, 1]) < 0.2]
    th, R = np.nan, 0.0
    if len(h) > 300:
        a = np.arctan2(h[:, 2], h[:, 0])
        c = np.exp(4j * a).sum()
        th, R = float(np.angle(c) / 4), float(np.abs(c) / len(h))
    # ---- floor height measurement ----
    pw = pc @ R0.T + p0
    fy, fn = np.nan, 0
    if floor_guess is not None:
        near = pw[np.abs(pw[:, 1] - floor_guess) < 0.12, 1]
        if len(near) > 200:
            fy, fn = robust_mode_1d(near, 0.005, 0.015)[0], len(near)
    pc_t = voxel_downsample(pc, cfg.per_frame_voxel_m).astype(np.float32)
    ec = edge_points(d, conf, K, cfg)
    return Keyframe(i, float(cap.t[i]), R0.copy(), p0.copy(), pc_t, ec, heading_of(R0), th, R, fy, fn)


def estimate_floor_guess(cap, kf_idx, cfg, n=40):
    sel = kf_idx[np.linspace(0, len(kf_idx) - 1, min(n, len(kf_idx))).astype(int)]
    ys = []
    for i in sel:
        d = cap.depth_m(i)
        pc, _, _ = backproject(d, cap.K_depth(i), cap.conf(i), 0.3, 4.0, cfg.min_confidence, cfg.edge_rel_thresh)
        pw = pc @ cap.T_c2w[i, :3, :3].T + cap.T_c2w[i, :3, 3]
        ys.append(pw[::3, 1])
    y = np.concatenate(ys)
    lo = np.percentile(y, 0.5)
    # floor = lowest *strong* peak (not the lowest outlier): peak holding >40% of max bin
    edges = np.arange(lo, np.percentile(y, 99.5), 0.02)
    h, e = np.histogram(y, bins=edges)
    strong = np.where(h >= 0.4 * h.max())[0]
    k = strong[0]
    return float(robust_mode_1d(y[np.abs(y - 0.5 * (e[k] + e[k + 1])) < 0.1], 0.005, 0.02)[0])


def build_keyframes(cap, cfg):
    kf_idx = select_keyframes(cap, cfg)
    floor_guess = estimate_floor_guess(cap, kf_idx, cfg)
    kfs = []
    for i in kf_idx:
        k = process_keyframe(cap, i, cfg, floor_guess)
        if k is not None:
            kfs.append(k)
    log.info("keyframes: %d (from %d depth frames), floor guess y=%.3f", len(kfs), int(cap.has_depth.sum()), floor_guess)
    return kfs, floor_guess


def world_points(kfs, psi=None, pos=None, voxel=None, band=None):
    """Fuse keyframes with (optionally corrected) yaw/position. psi/pos None => original poses."""
    out = []
    for j, k in enumerate(kfs):
        if psi is None:
            Rw = k.R0; p = k.p0
        else:
            Rw = rot_y(psi[j] - k.psi0) @ k.R0; p = pos[j]
        P = k.pts_cam.astype(np.float64) @ Rw.T + p
        if band is not None:
            P = P[(P[:, 1] > band[0]) & (P[:, 1] < band[1])]
        out.append(P)
    P = np.vstack(out)
    return voxel_downsample(P, voxel) if voxel else P


def world_edge_points(kfs, psi=None, pos=None):
    out = []
    for j, k in enumerate(kfs):
        if len(k.edge_cam) == 0:
            continue
        if psi is None:
            Rw, p = k.R0, k.p0
        else:
            Rw, p = rot_y(psi[j] - k.psi0) @ k.R0, pos[j]
        out.append(k.edge_cam.astype(np.float64) @ Rw.T + p)
    return np.vstack(out) if out else np.zeros((0, 3))
