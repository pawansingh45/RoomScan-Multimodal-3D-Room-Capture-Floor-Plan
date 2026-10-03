"""LiDAR tier: Record3D-style depth + poses + intrinsics -> plan."""
import time, logging
import numpy as np

from ..config import Config
from ..io.record3d import load_record3d, detect_pose_convention, apply_pose_convention
from ..lidar.frontend import build_keyframes, world_points, world_edge_points
from ..lidar.posegraph import optimise
from ..layout.rooms import build_rasters, segment_rooms, extract_rooms
from ..plan import assemble_plan
from ..uncertainty import IntervalModel

log = logging.getLogger("roomscan")


def wall_sharpness(P2, phi, res=0.01):
    """Ground-truth-free drift metric: fraction of wall points within 1.5 cm of the 14 strongest wall lines per axis.
    Drift smears walls (double walls, thick walls); correction sharpens them."""
    c, s = np.cos(phi), np.sin(phi)
    Q = np.c_[P2[:, 0] * c + P2[:, 1] * s, -P2[:, 0] * s + P2[:, 1] * c]
    tot = 0.0
    for ax in (0, 1):
        h, _ = np.histogram(Q[:, ax], bins=np.arange(Q[:, ax].min(), Q[:, ax].max() + res, res))
        hs = np.convolve(h, np.ones(3), "same").astype(float)
        hh = hs.copy(); pk = 0.0
        for _ in range(14):
            k = int(hh.argmax()); pk += hs[k]; hh[max(0, k - 15):k + 16] = 0
        tot += pk
    return float(tot / (2 * len(P2)))


def run_lidar(root, cfg: Config, drift_mode=None, keep_cloud=False, tier="lidar", extra_diag=None, device_class=None):
    t0 = time.time(); timing = {}
    mode = drift_mode or cfg.drift_mode
    cap = load_record3d(root)
    name, scores = detect_pose_convention(cap)
    apply_pose_convention(cap, name)
    timing["load"] = time.time() - t0; t1 = time.time()
    kfs, floor_guess = build_keyframes(cap, cfg)
    timing["keyframes"] = time.time() - t1; t1 = time.time()
    pg = optimise(kfs, floor_guess, cfg, mode)
    pg["info"]["rejected"] = False
    if mode != "off":
        # "do no harm" guard: a pose graph can be confidently wrong (bad priors, false loop closures). Accept the
        # correction only if a ground-truth-free map-sharpness metric (walls get thinner when poses improve) does not degrade.
        pg0 = optimise(kfs, floor_guess, cfg, "off")
        band = (floor_guess + 0.4, floor_guess + 1.8)
        P_off = world_points(kfs, pg0["psi"], pg0["pos"], 0.02, band=band)
        P_on = world_points(kfs, pg["psi"], pg["pos"], 0.02, band=band)
        ph = pg["phi"] if np.isfinite(pg["phi"]) else 0.0
        s_off = wall_sharpness(P_off[:, [0, 2]], ph) if len(P_off) > 1000 else 0.0
        s_on = wall_sharpness(P_on[:, [0, 2]], ph) if len(P_on) > 1000 else 0.0
        pg["info"].update(sharp_off=s_off, sharp_on=s_on)
        if s_on < s_off - 0.003:
            log.warning("pose-graph correction REJECTED (sharpness %.3f -> %.3f); keeping odometry poses", s_off, s_on)
            keep_phi = pg["phi"]
            pg = pg0; pg["info"]["rejected"] = True; pg["info"]["mode"] = "off(rejected %s)" % mode
            pg["phi"] = keep_phi if np.isfinite(keep_phi) else np.nan
    timing["posegraph"] = time.time() - t1; t1 = time.time()
    P = world_points(kfs, pg["psi"], pg["pos"], cfg.global_voxel_m)
    phi = pg["phi"]
    if not np.isfinite(phi):                       # drift_mode == off: still need a global Manhattan angle
        th = np.array([k.man_theta for k in kfs]); R = np.array([k.man_R for k in kfs]); ok = np.isfinite(th) & (R > 0.45)
        phi = float(np.angle(np.sum(R[ok] * np.exp(4j * th[ok]))) / 4) if ok.any() else 0.0
    cam_xz = pg["pos"][:, [0, 2]]
    E = world_edge_points(kfs, pg["psi"], pg["pos"])
    rs = build_rasters(P, floor_guess, phi, cam_xz, cfg, edge_pts=E)
    labels, virt, dist = segment_rooms(rs, cfg)
    rooms = extract_rooms(rs, labels, cfg)
    timing["layout"] = time.time() - t1; t1 = time.time()
    im = IntervalModel(level=cfg.ci_level)
    plan = assemble_plan(rs, labels, rooms, tier, cfg, im, phi)
    timing["assemble"] = time.time() - t1
    wall_xz = P[(P[:, 1] > floor_guess + 0.4) & (P[:, 1] < floor_guess + 1.8)][:, [0, 2]]
    plan["capture"] = dict(tier=tier, status="ok" if plan["rooms"] else "insufficient", source=str(root), n_poses=int(cap.n),
                           n_depth_frames=int(cap.has_depth.sum()), n_keyframes=len(kfs),
                           duration_s=float(cap.t[-1] - cap.t[0]), device_class=device_class or "LiDAR Pro-class iPhone (Record3D-style export)")
    plan["diagnostics"] = dict(
        pose_convention=name, pose_convention_scores=scores, floor_y_world=float(floor_guess),
        drift=dict(requested_mode=mode, **pg["info"]),
        wall_sharpness=wall_sharpness(wall_xz, phi) if len(wall_xz) > 1000 else None,
        timing_s={k: round(v, 2) for k, v in timing.items()}, total_s=round(time.time() - t0, 2),
        config=cfg.to_dict(), intervals_calibrated=bool(im.p["calibrated"]),
        notes=["Intervals are engineering priors until `roomscan calibrate` is run on laser-measured ground truth."], **(extra_diag or {}))
    extra = None
    if keep_cloud:
        extra = dict(P=P, rs=rs, labels=labels, floor_y=floor_guess, phi=phi, kfs=kfs, pg=pg, cap=cap)
    return plan, extra
