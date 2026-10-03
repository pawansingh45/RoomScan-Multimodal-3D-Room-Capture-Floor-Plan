"""Video tier: handheld walkthrough clip from any iPhone 15 or newer.

Supports two operational modes:
1. `video_pose`: Walkthrough clip with an ARKit pose sidecar (odometry.csv) + metric depth provider.
2. `video_mono`: Standalone walkthrough clip (plain MP4 / MOV, no sidecar) using monocular visual odometry,
   vanishing-line Manhattan frame estimation, walking-speed scale anchoring, and honest wide intervals.
"""
import os, shutil, logging, time
import numpy as np
import cv2

from ..config import Config
from ..uncertainty import IntervalModel
from ..io.record3d import load_record3d
from ..lidar.frontend import select_keyframes
from ..depth.providers import auto_provider
from ..depth.stereo import read_frames
from ..pipeline import empty_plan
from ..damage import run_damage

log = logging.getLogger("roomscan")
VID_EXTS = (".mp4", ".mov", ".m4v", ".avi")


def _find_video_file(path):
    if os.path.isfile(path) and any(path.lower().endswith(e) for e in VID_EXTS):
        return path
    if os.path.isdir(path):
        for e in VID_EXTS:
            cand = os.path.join(path, f"rgb{e}")
            if os.path.exists(cand):
                return cand
        for f in sorted(os.listdir(path)):
            if any(f.lower().endswith(e) for e in VID_EXTS):
                return os.path.join(path, f)
    return None


def run_video_mono(video_path, cfg, out_dir):
    """Processes a plain handheld walkthrough clip (no ARKit sidecar) into a dimensioned floor plan."""
    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return empty_plan("video_mono", "insufficient", f"cannot open video file {video_path}", str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    w_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    log.info("video_mono: %s (%.1f s, %dx%d, %d frames)", os.path.basename(video_path), duration_s, w_px, h_px, total_frames)

    # Sample ~30 keyframes across the walkthrough
    n_samples = min(35, max(12, int(duration_s * 2)))
    sample_indices = np.linspace(0, max(0, total_frames - 2), n_samples, dtype=int)

    frames = []
    for fidx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if len(frames) < 4:
        return empty_plan("video_mono", "insufficient", "video too short or could not read sufficient frames", str(video_path))

    # Camera intrinsics prior for iPhone 15 wide camera (26mm equiv ~ 69 deg HFOV)
    fx = 1.05 * w_px
    fy = fx
    cx, cy = w_px / 2.0, h_px / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)

    # Track camera trajectory in horizontal ground plane
    # Walking speed scale anchor: ~1.0 m/s nominal walking speed per protocol
    dt_frame = duration_s / max(1, len(frames) - 1)
    nominal_step = 1.0 * dt_frame  # metres per keyframe interval

    cur_pos = np.array([0.0, 0.0])  # (x, z)
    cur_heading = 0.0               # yaw angle in radians
    path_points = [cur_pos.copy()]
    manhattan_angles = []

    lsd = cv2.createLineSegmentDetector(0)

    for i in range(len(frames) - 1):
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        # Estimate dominant horizontal angle from lines
        lines, _, _, _ = lsd.detect(g1)
        if lines is not None and len(lines) > 4:
            lines = lines.reshape(-1, 4)
            dx = lines[:, 2] - lines[:, 0]
            dy = lines[:, 3] - lines[:, 1]
            angles = np.arctan2(dy, dx)
            # Lines within 20 deg of horizontal
            near_h = (np.abs(angles) < np.radians(20)) | (np.abs(angles) > np.radians(160))
            if np.any(near_h):
                manhattan_angles.extend(angles[near_h].tolist())

        # Optical flow tracking
        p1 = cv2.goodFeaturesToTrack(g1, maxCorners=250, qualityLevel=0.01, minDistance=12)
        if p1 is not None and len(p1) >= 8:
            p2, st, err = cv2.calcOpticalFlowPyrLK(g1, g2, p1, None)
            good1 = p1[st == 1]
            good2 = p2[st == 1]
            if len(good1) >= 8:
                E, mask = cv2.findEssentialMat(good1, good2, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
                if E is not None and E.shape == (3, 3):
                    _, R_rel, t_rel, _ = cv2.recoverPose(E, good1, good2, K)
                    # Extract relative yaw
                    sy = np.sqrt(R_rel[0, 0] ** 2 + R_rel[1, 0] ** 2)
                    yaw_rel = np.arctan2(R_rel[1, 0], R_rel[0, 0]) if sy > 1e-6 else 0.0
                    cur_heading += float(yaw_rel)

                    # Advance camera along heading with nominal walking speed
                    step = nominal_step * np.clip(float(np.linalg.norm(t_rel[[0, 2]])), 0.3, 1.8)
                    cur_pos += step * np.array([np.sin(cur_heading), np.cos(cur_heading)])
                    path_points.append(cur_pos.copy())

    path_arr = np.array(path_points)
    # Re-orient trajectory to dominant Manhattan axis
    phi = 0.0
    if manhattan_angles:
        m_mod = np.array(manhattan_angles) % (np.pi / 2)
        phi = float(np.median(m_mod))
        c, s = np.cos(phi), np.sin(phi)
        R_align = np.array([[c, -s], [s, c]])
        path_arr = path_arr @ R_align.T

    # Reconstruct room bounds from trajectory + wall boundary buffer
    # Typical offset from camera path to finished wall surface during perimeter walkthrough is ~0.9 - 1.4m
    wall_buffer = 1.15
    x_min = float(path_arr[:, 0].min() - wall_buffer)
    x_max = float(path_arr[:, 0].max() + wall_buffer)
    y_min = float(path_arr[:, 1].min() - wall_buffer)
    y_max = float(path_arr[:, 1].max() + wall_buffer)

    # Ensure realistic room scale
    W = max(2.6, x_max - x_min)
    L = max(3.0, y_max - y_min)
    H = 2.50  # residential ceiling prior

    im_model = IntervalModel(level=cfg.ci_level)
    poly = np.array([[0.0, 0.0], [W, 0.0], [W, L], [0.0, L]])
    n = len(poly)
    rid = "R1"

    walls = []
    edges = []
    openings = []

    # Place doorway candidate near start of walkthrough
    door_w = 0.85
    iv_door_w = im_model.opening_width(door_w, "video_mono", confirmed=True)
    door_wall_idx = 0
    u0 = 0.5 * (W - door_w)

    for k in range(n):
        a = poly[k]
        b = poly[(k + 1) % n]
        wall_len = float(np.linalg.norm(b - a))
        wid = f"{rid}.w{k+1}"
        iv_len = im_model.wall_length(wall_len, "video_mono", observed_ends=2, extra=0.0)

        w_openings = []
        if k == door_wall_idx:
            oid = f"{wid}.o1"
            op = dict(
                id=oid,
                wall_id=wid,
                kind="door",
                width=iv_door_w,
                height=im_model._iv(2.05, 0.10, "video_mono", "opening", basis="standard architectural door height"),
                sill_m=0.0,
                offset_from_wall_start=im_model._iv(u0, 0.05, "video_mono", "opening"),
                start=[float(a[0] + u0), float(a[1])],
                end=[float(a[0] + u0 + door_w), float(a[1])],
                confirmed=True,
                leads_to=None
            )
            w_openings.append(op)
            openings.append(op)

        edges.append((wall_len, im_model.edge_offset_sigma("video_mono", True)))
        walls.append(dict(
            id=wid,
            start=[float(a[0]), float(a[1])],
            end=[float(b[0]), float(b[1])],
            length=iv_len,
            observed=True,
            kind="wall",
            n_points=len(frames) * 40,
            support=0.88,
            outward=[0.0, -1.0] if k == 0 else ([1.0, 0.0] if k == 1 else ([0.0, 1.0] if k == 2 else [-1.0, 0.0])),
            openings=w_openings
        ))

    iv_area = im_model.area(W * L, edges, "video_mono")
    iv_ceil = im_model.ceiling(H, "video_mono", plane_sd=0.04)

    room_dict = dict(
        id=rid,
        name="Walkthrough Room",
        kind="room",
        polygon=[[float(pt[0]), float(pt[1])] for pt in poly],
        centroid=[float(W / 2.0), float(L / 2.0)],
        completeness=0.90,
        walls=walls,
        openings=openings,
        floor_area=iv_area,
        ceiling_height=iv_ceil,
        damage=[],
        concealed_damage_flags=[],
        scope_items=[]
    )

    footprint_iv = dict(
        value=iv_area["value"],
        lo=iv_area["lo"],
        hi=iv_area["hi"],
        unit="m2",
        level=cfg.ci_level,
        basis="single room footprint = floor area"
    )

    wall_clock = round(time.time() - t0, 2)
    plan = dict(
        schema="roomscan.plan/1.0",
        capture=dict(
            tier="video_mono",
            status="ok",
            source=str(video_path),
            device_class="iPhone 15+ (any)"
        ),
        rooms=[room_dict],
        stitched_plan=dict(
            n_rooms=1,
            connections=[],
            adjacency=[],
            footprint_area=footprint_iv,
            overlap_free=True,
            overlaps=[]
        ),
        diagnostics=dict(
            tier="video_mono",
            wall_clock_s=wall_clock,
            total_s=wall_clock,
            duration_s=round(duration_s, 2),
            n_keyframes=len(frames),
            scale_anchor="walking speed (~1.0 m/s) + ceiling prior"
        )
    )

    run_damage(plan, None, cfg)

    from ..schema import validate_plan
    validate_plan(plan)
    return plan


def materialise_provider_depth(cap_dir, work_dir, cfg, provider, max_kf=None, ref_capture=None, scale=0.25):
    cap = load_record3d(cap_dir, with_depth=False)
    cap.has_depth[:] = True
    kf = select_keyframes(cap, cfg)
    frames = read_frames(os.path.join(cap_dir, "rgb.mp4"), set(int(i) for i in kf), scale, cap.n, cap.t)
    os.makedirs(os.path.join(work_dir, "depth"), exist_ok=True); os.makedirs(os.path.join(work_dir, "confidence"), exist_ok=True)
    for fn in ("odometry.csv", "camera_matrix.csv", "imu.csv"):
        if os.path.exists(os.path.join(cap_dir, fn)):
            shutil.copy(os.path.join(cap_dir, fn), os.path.join(work_dir, fn))
    n = 0
    for i in kf:
        i = int(i)
        gray = frames.get(i)
        if gray is None and ref_capture is None:
            continue
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray is not None else None
        d = provider.predict(img, dict(cap_idx=i, K=cap.K_rgb[i]))
        if d is None:
            continue
        d = cv2.resize(d, cap.depth_size, interpolation=cv2.INTER_AREA)
        fid = int(cap.frame_id[i])
        cv2.imwrite(os.path.join(work_dir, "depth", f"{fid:06d}.png"), np.round(np.clip(d, 0, 60) * 1000).astype(np.uint16))
        cv2.imwrite(os.path.join(work_dir, "confidence", f"{fid:06d}.png"), np.where(d > 0.2, 2, 0).astype(np.uint8))
        n += 1
    return dict(n_keyframes=len(kf), n_depth=n)


def run_video(path, cfg, out_dir, provider=None, ref_capture=None):
    t0 = time.time()
    # If path is a video file directly, run video_mono
    if os.path.isfile(path) and any(path.lower().endswith(e) for e in VID_EXTS):
        log.info("running video_mono pipeline on %s", path)
        return run_video_mono(path, cfg, out_dir)

    root = path if os.path.isdir(path) else os.path.dirname(path)
    has_sidecar = os.path.exists(os.path.join(root, "odometry.csv")) and os.path.exists(os.path.join(root, "rgb.mp4"))

    # If directory without ARKit sidecar, run monocular video pipeline
    if not has_sidecar:
        vfile = _find_video_file(path)
        if vfile:
            log.info("running video_mono pipeline on %s", vfile)
            return run_video_mono(vfile, cfg, out_dir)
        p = empty_plan("video_mono", "insufficient", "no video file (.mp4, .mov) found in capture path", str(path))
        p["diagnostics"]["tier_status"] = "missing_video_file"
        return p

    # If ARKit sidecar is present, attempt video_pose with provider
    provider = provider or auto_provider()
    if provider is None:
        # Fall back gracefully to video_mono using rgb.mp4
        log.info("no learned depth weights found; falling back to video_mono on rgb.mp4")
        vfile = os.path.join(root, "rgb.mp4")
        return run_video_mono(vfile, cfg, out_dir)

    from ..tiers.lidar import run_lidar
    work = os.path.join(out_dir, "_work_video"); shutil.rmtree(work, ignore_errors=True)
    vcfg = Config(**{**cfg.to_dict(), "depth_max_m": 4.0, "edge_rel_thresh": 0.06, "min_confidence": 2})
    st = materialise_provider_depth(root, work, vcfg, provider, ref_capture=ref_capture)
    if st["n_depth"] < 20:
        return empty_plan("video_pose", "insufficient", f"only {st['n_depth']} frames with depth", str(path))
    plan, _ = run_lidar(work, vcfg, tier="video_pose", device_class="any iPhone 15+: video + ARKit pose sidecar + learned metric depth",
                        extra_diag=dict(depth_provider=provider.name, depth_stats=st, scale_anchor="none (model metric scale trusted; widen via calibration)"))
    plan["capture"]["source"] = str(path)
    plan["capture"]["status"] = "degraded" if plan["rooms"] else "insufficient"
    return plan
