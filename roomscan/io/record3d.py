"""Loader for Record3D-style LiDAR captures (the format of the supplied samples).

Layout:  <root>/rgb.mp4, depth/NNNNNN.png (uint16, millimetres, 256x192), confidence/NNNNNN.png (0..2),
         odometry.csv (timestamp, frame, x,y,z, qx,qy,qz,qw, fx,fy,cx,cy), imu.csv, camera_matrix.csv.

Everything is synchronised by *timestamp/frame id*, never by array index: the video is variable frame rate
and frames can be dropped (verified on the samples: gaps up to 50 ms).
"""
import os, logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import cv2

from ..geom import quat_to_R

log = logging.getLogger("roomscan")


@dataclass
class LidarCapture:
    root: str
    T_c2w: np.ndarray            # (N,4,4) camera->world in the standard CV camera frame (x right, y down, z fwd)
    t: np.ndarray                # (N,) seconds from start
    frame_id: np.ndarray         # (N,) int frame ids (file names)
    K_rgb: np.ndarray            # (N,3,3) per-frame intrinsics at RGB resolution
    rgb_size: tuple              # (W,H)
    depth_size: tuple            # (W,H)
    has_depth: np.ndarray        # (N,) bool
    imu: pd.DataFrame = None
    pose_convention: str = "cv_c2w"
    convention_score: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.t)

    def K_depth(self, i):
        K = self.K_rgb[i].copy()
        K[0] *= self.depth_size[0] / self.rgb_size[0]
        K[1] *= self.depth_size[1] / self.rgb_size[1]
        return K

    def depth_m(self, i):
        p = os.path.join(self.root, "depth", f"{int(self.frame_id[i]):06d}.png")
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        return None if d is None else d.astype(np.float32) / 1000.0

    def conf(self, i):
        p = os.path.join(self.root, "confidence", f"{int(self.frame_id[i]):06d}.png")
        return cv2.imread(p, cv2.IMREAD_UNCHANGED)

    def video_path(self):
        p = os.path.join(self.root, "rgb.mp4")
        return p if os.path.exists(p) else None


def is_record3d(root):
    return os.path.isfile(os.path.join(root, "odometry.csv")) and os.path.isdir(os.path.join(root, "depth"))


def _probe_video_size(path, fallback):
    try:
        cap = cv2.VideoCapture(path)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        pass
    return fallback


def load_record3d(root, with_depth=True):
    odo = pd.read_csv(os.path.join(root, "odometry.csv"), skipinitialspace=True)
    odo.columns = [c.strip() for c in odo.columns]
    odo = odo.sort_values("timestamp").reset_index(drop=True)
    frame_id = odo["frame"].astype(int).values
    n = len(odo)
    Rs = np.array([quat_to_R(r.qx, r.qy, r.qz, r.qw) for r in odo.itertuples()])
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rs
    T[:, :3, 3] = odo[["x", "y", "z"]].values
    K0 = None
    kp = os.path.join(root, "camera_matrix.csv")
    if os.path.exists(kp):
        K0 = np.loadtxt(kp, delimiter=",")
    K = np.tile(np.eye(3), (n, 1, 1))
    if all(c in odo.columns for c in ("fx", "fy", "cx", "cy")) and not odo[["fx", "cx"]].isna().any().any():
        K[:, 0, 0] = odo.fx.values; K[:, 1, 1] = odo.fy.values      # ARKit refines intrinsics per frame
        K[:, 0, 2] = odo.cx.values; K[:, 1, 2] = odo.cy.values
    elif K0 is not None:
        K[:] = K0
    else:
        raise ValueError("no intrinsics found (need per-frame fx..cy in odometry.csv or camera_matrix.csv)")
    vpath = os.path.join(root, "rgb.mp4")
    fb = (int(round(K[0, 0, 2] * 2)), int(round(K[0, 1, 2] * 2)))
    rgb_size = _probe_video_size(vpath, fb) if os.path.exists(vpath) else fb
    ddir = os.path.join(root, "depth")
    avail, dsize = set(), (256, 192)
    if os.path.isdir(ddir):
        names = sorted(x for x in os.listdir(ddir) if x.endswith(".png") and x[:-4].isdigit())
        avail = {int(x[:-4]) for x in names}
        if names:
            d0 = cv2.imread(os.path.join(ddir, names[0]), cv2.IMREAD_UNCHANGED)
            if d0 is not None:
                dsize = (d0.shape[1], d0.shape[0])
    has_depth = np.array([int(f) in avail for f in frame_id]) if with_depth else np.zeros(n, bool)
    imu = pd.read_csv(os.path.join(root, "imu.csv"), skipinitialspace=True) if os.path.exists(os.path.join(root, "imu.csv")) else None
    cap = LidarCapture(root=root, T_c2w=T, t=odo.timestamp.values - odo.timestamp.values[0], frame_id=frame_id,
                       K_rgb=K, rgb_size=rgb_size, depth_size=dsize, has_depth=has_depth, imu=imu)
    log.info("loaded %s: %d poses, %d with depth, rgb %s depth %s", root, n, int(has_depth.sum()), rgb_size, dsize)
    return cap


def backproject(d, K, conf=None, dmin=0.25, dmax=4.5, min_conf=2, edge_rel=0.04):
    """Depth image -> camera-frame points (N,3) plus pixel coords. Drops low-confidence and 'flying' edge pixels."""
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    m = (d > dmin) & (d < dmax)
    if conf is not None:
        m &= conf >= min_conf
    if edge_rel:
        # flying pixels: strong depth discontinuity to any 4-neighbour
        dd = np.zeros_like(d, bool)
        for a, b in ((d[:, 1:], d[:, :-1]), (d[1:], d[:-1])):
            bad = np.abs(a - b) > edge_rel * np.maximum(a, b)
            if a.shape[0] == H and a.shape[1] == W - 1:
                dd[:, 1:] |= bad; dd[:, :-1] |= bad
            else:
                dd[1:] |= bad; dd[:-1] |= bad
        m &= ~dd
    z = d[m]
    x = (u[m] - K[0, 2]) * z / K[0, 0]
    y = (v[m] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1), (u[m], v[m]), m


# ---------------------------------------------------------------- pose-convention detection
def detect_pose_convention(cap, n_frames=30):
    """Decide how odometry quaternions map to the CV camera frame by *geometry*, not assumption.

    Under the right convention the floor collapses to one height: a razor-sharp histogram peak along the
    gravity axis. Candidates: camera-axis flips (CV / OpenGL) x (R, R^T). Returns (name, scores).
    """
    idx = np.where(cap.has_depth)[0]
    if len(idx) == 0:
        return "cv_c2w", {}
    idx = idx[np.linspace(0, len(idx) - 1, min(n_frames, len(idx))).astype(int)]
    cands = {"cv_c2w": (np.eye(3), False), "gl_c2w": (np.diag([1, -1, -1.0]), False),
             "cv_w2c": (np.eye(3), True), "gl_w2c": (np.diag([1, -1, -1.0]), True)}
    frames = []
    for i in idx:
        d = cap.depth_m(i)
        if d is None:
            continue
        pc, _, _ = backproject(d, cap.K_depth(i), cap.conf(i), 0.3, 4.0, 2, 0.0)
        frames.append((i, pc[::2]))
    scores = {}
    for name, (flip, inv) in cands.items():
        pts = []
        for i, pc in frames:
            R, t = cap.T_c2w[i, :3, :3], cap.T_c2w[i, :3, 3]
            if inv:
                R, t = R.T, -R.T @ t
            pts.append((pc @ flip.T) @ R.T + t)
        P = np.vstack(pts)
        best = 0.0
        for ax in range(3):
            lo, hi = np.percentile(P[:, ax], [0.5, 99.5])
            h, _ = np.histogram(P[:, ax], bins=np.arange(lo, hi + 0.02, 0.02))
            if len(h):
                best = max(best, h.max() / len(P))
        scores[name] = float(best)
    return max(scores, key=scores.get), scores


def apply_pose_convention(cap, name):
    """Normalise poses in place to: CV camera frame, camera->world."""
    flip = np.diag([1, -1, -1.0]) if name.startswith("gl") else np.eye(3)
    for i in range(cap.n):
        R, t = cap.T_c2w[i, :3, :3].copy(), cap.T_c2w[i, :3, 3].copy()
        if name.endswith("w2c"):
            R, t = R.T, -R.T @ t
        cap.T_c2w[i, :3, :3] = R @ flip
        cap.T_c2w[i, :3, 3] = t
    cap.pose_convention = name
    return cap
