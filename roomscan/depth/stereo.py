"""Metric depth from video + metric poses (ARKit sidecar): classical two-view stereo between keyframe pairs.

For keyframe i pick a partner j with a 6-25 cm baseline and small relative rotation, rectify with the KNOWN relative pose
(cv2.stereoRectify), run SGBM, triangulate in the rectified frame, rotate back to camera i, and splat into a depth image at the
LiDAR depth resolution (256x192). No network / GPU / learned weights. Valid only where there is texture: untextured walls
give holes (never invented depth); confidence 2 = left-right-consistent, unique match.
"""
import os, shutil, logging
import numpy as np
import cv2

log = logging.getLogger("roomscan")


def read_frames(video_path, wanted, scale=0.25, n_poses=None, times=None):
    """Sequentially decode and keep frames whose index is in `wanted` (video frame n == odometry row n when counts match)."""
    cap = cv2.VideoCapture(video_path)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = sorted(set(int(w) for w in wanted))
    if n_poses is not None and nfr != n_poses and times is not None:        # variable frame rate / dropped frames: map by time
        fps_t = np.array([cap.get(cv2.CAP_PROP_POS_MSEC)])
        mapping = {}
        dur = nfr / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
        vt = np.linspace(0, dur, nfr)
        for w in wanted:
            mapping[w] = int(np.argmin(np.abs(vt - times[w])))
    else:
        mapping = {w: w for w in wanted}
    inv = {}
    for w, f in mapping.items():
        inv.setdefault(f, []).append(w)
    out, idx, last = {}, 0, max(inv) if inv else -1
    while idx <= last:
        ok = cap.grab()
        if not ok:
            break
        if idx in inv:
            ok, fr = cap.retrieve()
            if ok:
                g = cv2.cvtColor(cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                for w in inv[idx]:
                    out[w] = g
        idx += 1
    cap.release()
    return out


def pick_partner(T, i, cands, bmin=0.06, bmax=0.25, max_rot_deg=10, target=0.12, exclude=()):
    best, bs = None, 1e9
    for j in cands:
        if j in exclude:
            continue
        if j == i:
            continue
        b = np.linalg.norm(T[j, :3, 3] - T[i, :3, 3])
        if not (bmin <= b <= bmax):
            continue
        Rr = T[i, :3, :3].T @ T[j, :3, :3]
        ang = np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1)))
        if ang > max_rot_deg:
            continue
        s = abs(b - target) + 0.002 * ang
        if s < bs:
            best, bs = j, s
    return best


def _stereo_points(gl, gr, Kl, Kr, Tl, Tr, z_range, min_disp_px):
    """3D points in the LEFT camera frame from a pair where the right camera is physically to the right (disparity>0)."""
    h, w = gl.shape
    Rrel = Tr[:3, :3].T @ Tl[:3, :3]
    Trel = Tr[:3, :3].T @ (Tl[:3, 3] - Tr[:3, 3])
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(Kl, None, Kr, None, (w, h), Rrel, Trel, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if P2[0, 3] > 0:                       # right camera ended up on the left: caller must swap
        return None, None
    m1 = cv2.initUndistortRectifyMap(Kl, None, R1, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(Kr, None, R2, P2, (w, h), cv2.CV_32FC1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    a = cv2.remap(clahe.apply(gl), *m1, cv2.INTER_LINEAR); b = cv2.remap(clahe.apply(gr), *m2, cv2.INTER_LINEAR)
    f = float(P1[0, 0]); B = float(abs(P2[0, 3] / P2[0, 0]))
    dmax = int(np.ceil(f * B / z_range[0] / 16.0) * 16)
    dmax = max(32, min(dmax, 192))
    sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=dmax, blockSize=7, P1=8 * 49, P2=32 * 49, disp12MaxDiff=1,
                               uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sg.compute(a, b).astype(np.float32) / 16.0
    ok = disp > min_disp_px
    if ok.sum() < 200:
        return None, None
    pts = cv2.reprojectImageTo3D(disp, Q)[ok]
    X = pts @ R1                            # rectified -> original left camera frame
    g = (X[:, 2] > z_range[0]) & (X[:, 2] < z_range[1])
    return X[g], dict(f=f, B=B, valid=float(ok.mean()))


def stereo_depth(gi, gj, Ki, Kj, Ti, Tj, out_size=(256, 192), z_range=(0.4, 4.0), min_disp_px=2.0):
    """Depth image for camera i at out_size from (gi,gj) grayscale pair. Ki,Kj at the grayscale resolution."""
    h, w = gi.shape
    X, info = _stereo_points(gi, gj, Ki, Kj, Ti, Tj, z_range, min_disp_px)
    if X is None:                           # partner is on the left: make it the left camera, then express points in camera i
        Xj, info = _stereo_points(gj, gi, Kj, Ki, Tj, Ti, z_range, min_disp_px)
        if Xj is None:
            return None
        Tji = np.linalg.inv(Ti) @ Tj        # camera j -> camera i
        X = Xj @ Tji[:3, :3].T + Tji[:3, 3]
        X = X[(X[:, 2] > z_range[0]) & (X[:, 2] < z_range[1])]
    if len(X) < 200:
        return None
    sx, sy = out_size[0] / w, out_size[1] / h
    u = (Ki[0, 0] * X[:, 0] / X[:, 2] + Ki[0, 2]) * sx
    v = (Ki[1, 1] * X[:, 1] / X[:, 2] + Ki[1, 2]) * sy
    ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
    inb = (ui >= 0) & (ui < out_size[0]) & (vi >= 0) & (vi < out_size[1])
    # per-pixel MEDIAN depth of the points that land in it (min() would bias near)
    key = vi[inb] * out_size[0] + ui[inb]; z = X[inb, 2]
    o = np.lexsort((z, key)); key, z = key[o], z[o]
    uk, st, cn = np.unique(key, return_index=True, return_counts=True)
    med = z[st + cn // 2]
    depth = np.zeros(out_size[0] * out_size[1], np.float32)
    keep = cn >= 2
    depth[uk[keep]] = med[keep]
    return depth.reshape(out_size[1], out_size[0])


def consistent_depth(d1, d2, tol=0.04):
    """Multi-baseline consistency: keep pixels where two independent pairs agree within tol (aliasing differs per baseline)."""
    if d1 is None or d2 is None:
        return None
    ok = (d1 > 0) & (d2 > 0) & (np.abs(d1 - d2) < tol * 0.5 * (d1 + d2))
    if ok.sum() < 100:
        return None
    out = np.zeros_like(d1)
    out[ok] = 0.5 * (d1[ok] + d2[ok])
    return out


def materialise_depth(cap_dir, work_dir, cfg, max_pairs=None, progress=True):
    """Create a Record3D-layout work dir with stereo depth (depth/, confidence/) + copied poses. Returns stats dict."""
    from ..io.record3d import load_record3d, detect_pose_convention  # noqa
    from ..lidar.frontend import select_keyframes
    cap = load_record3d(cap_dir, with_depth=False)
    vp = os.path.join(cap_dir, "rgb.mp4")
    if not os.path.exists(vp):
        raise RuntimeError("video tier needs rgb.mp4")
    # pose convention: cannot be detected from depth here; Record3D export is cv_c2w (verified on the samples by geometry)
    cap.has_depth[:] = True
    kf = select_keyframes(cap, cfg)
    T = cap.T_c2w
    pairs = {}
    for i in kf:
        j = pick_partner(T, i, range(max(0, i - 40), min(cap.n, i + 41)))
        if j is not None:
            pairs[int(i)] = int(j)
    need = set(pairs) | set(pairs.values())
    frames = read_frames(vp, need, 0.25, cap.n, cap.t)
    os.makedirs(os.path.join(work_dir, "depth"), exist_ok=True); os.makedirs(os.path.join(work_dir, "confidence"), exist_ok=True)
    for fn in ("odometry.csv", "camera_matrix.csv", "imu.csv"):
        if os.path.exists(os.path.join(cap_dir, fn)):
            shutil.copy(os.path.join(cap_dir, fn), os.path.join(work_dir, fn))
    cov, n_ok = [], 0
    for k, (i, j) in enumerate(sorted(pairs.items())):
        if i not in frames or j not in frames:
            continue
        Ki = cap.K_rgb[i].copy(); Ki[:2] *= 0.25; Kj = cap.K_rgb[j].copy(); Kj[:2] *= 0.25
        d = stereo_depth(frames[i], frames[j], Ki, Kj, T[i], T[j], cap.depth_size)
        if d is None:
            continue
        fid = int(cap.frame_id[i])
        cv2.imwrite(os.path.join(work_dir, "depth", f"{fid:06d}.png"), np.round(d * 1000).astype(np.uint16))
        cv2.imwrite(os.path.join(work_dir, "confidence", f"{fid:06d}.png"), np.where(d > 0, 2, 0).astype(np.uint8))
        cov.append(float((d > 0).mean())); n_ok += 1
        if progress and k % 50 == 0:
            log.info("stereo %d/%d", k, len(pairs))
    vlink = os.path.join(work_dir, "rgb.mp4")
    if not os.path.exists(vlink):
        try:
            os.symlink(os.path.abspath(vp), vlink)
        except OSError:
            shutil.copy(vp, vlink)
    return dict(n_keyframes=len(kf), n_pairs=len(pairs), n_depth=n_ok, mean_valid_fraction=float(np.mean(cov)) if cov else 0.0)
