"""Synthetic Record3D-format capture of a known apartment (exact ground truth), with noise and drift.

Geometry is analytic: axis-aligned wall faces (with door cut-outs, lintels and jamb faces), floor, per-room ceiling.
Depth is ray-cast per pixel. Use: validate algorithms, unit tests, drift ablations, interval calibration priors.
Everything here is SYNTHETIC: results from it are labelled as such and never reported as real-world accuracy.
"""
import os, json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot


def default_apartment():
    """Three rooms + a connector (corridor). Units m. Rooms are rectangles [x0,x1]x[z0,z1]."""
    rooms = {
        "A": dict(rect=(0.0, 4.2, 0.0, 3.4), ceiling=2.55),     # living room
        "B": dict(rect=(0.0, 3.0, 3.6, 6.1), ceiling=2.45),     # bedroom
        "C": dict(rect=(4.4, 6.9, 0.0, 2.6), ceiling=2.40),     # bathroom-ish
        "H": dict(rect=(3.2, 4.2, 3.6, 6.1), ceiling=2.55),     # hallway/connector
    }
    # doors: (wall line orientation 'h' = along x at constant z, 'v' = along z at constant x), coord, span, height
    doors = [
        dict(id="A-B", orient="h", c=3.5, u0=0.6, u1=1.5, h=2.05),       # in wall z=3.5 between A and B (x 0.6..1.5)
        dict(id="A-H", orient="h", c=3.5, u0=3.35, u1=4.05, h=2.05),     # between A and H
        dict(id="A-C", orient="v", c=4.3, u0=0.8, u1=1.62, h=2.05),      # between A and C (z 0.8..1.62)
    ]
    return dict(rooms=rooms, doors=doors, wall_t=0.12, outer=(-0.1, 7.0, -0.1, 6.2))


def _faces(apt):
    """Return list of rectangles: dict(axis, c, a0,a1 (along tangent t), h0,h1) for vertical faces,
    plus horizontal planes handled separately. axis: 0 => plane x=c (tangent z), 1 => plane z=c (tangent x)."""
    t = apt["wall_t"]; H = 3.2
    rooms = apt["rooms"]
    # wall lines: each room rect edge = a wall slab centred 0.06 outside the interior rect
    lines = {}
    for rid, r in rooms.items():
        x0, x1, z0, z1 = r["rect"]
        lines.setdefault(("h", round(z0 - t / 2, 3)), []).append((x0 - t, x1 + t))
        lines.setdefault(("h", round(z1 + t / 2, 3)), []).append((x0 - t, x1 + t))
        lines.setdefault(("v", round(x0 - t / 2, 3)), []).append((z0 - t, z1 + t))
        lines.setdefault(("v", round(x1 + t / 2, 3)), []).append((z0 - t, z1 + t))
    faces = []
    for (orient, c), spans in lines.items():
        # merge spans
        spans = sorted(spans); merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            # doors on this slab (door centre lines are given as the wall centre coordinate c)
            ds = [d for d in apt["doors"] if d["orient"] == orient and abs(d["c"] - c) < 0.2 and d["u0"] >= a - 1e-6 and d["u1"] <= b + 1e-6]
            cuts = sorted([(d["u0"], d["u1"], d["h"]) for d in ds])
            axis = 1 if orient == "h" else 0
            for side in (-1, 1):
                cc = c + side * t / 2
                pos = a
                for u0, u1, hd in cuts:
                    if u0 > pos + 1e-6:
                        faces.append(dict(axis=axis, c=cc, a0=pos, a1=u0, h0=0, h1=H))
                    faces.append(dict(axis=axis, c=cc, a0=u0, a1=u1, h0=hd, h1=H))      # lintel
                    pos = u1
                if b > pos + 1e-6:
                    faces.append(dict(axis=axis, c=cc, a0=pos, a1=b, h0=0, h1=H))
            for u0, u1, hd in cuts:                                                      # jamb faces
                for uu in (u0, u1):
                    faces.append(dict(axis=1 - axis, c=uu, a0=c - t / 2, a1=c + t / 2, h0=0, h1=hd))
    return faces


def _ceiling_at(apt, x, z):
    h = np.full(x.shape, 3.2)
    for r in apt["rooms"].values():
        x0, x1, z0, z1 = r["rect"]
        m = (x >= x0 - 0.06) & (x <= x1 + 0.06) & (z >= z0 - 0.06) & (z <= z1 + 0.06)
        h[m] = np.minimum(h[m], r["ceiling"]) if True else h[m]
    return h


def render_depth(apt, faces, R, p, K, size=(256, 192), rng=None, noise=(0.003, 0.004)):
    W, Hh = size
    v, u = np.mgrid[0:Hh, 0:W]
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
    d = d_cam @ R.T                                   # world ray directions
    o = p
    tmin = np.full(len(d), np.inf)
    # floor (y=0 in "room" convention; we use world Y-up with floor at y = -1.4 later; here floor at y=0, up=+y)
    with np.errstate(divide="ignore", invalid="ignore"):
        for plane_y in (0.0,):
            t = (plane_y - o[1]) / d[:, 1]
            ok = t > 0.05
            tmin = np.where(ok & (t < tmin), t, tmin)
        # ceilings: per-location
        for cy in sorted({r["ceiling"] for r in apt["rooms"].values()} | {3.2}):
            t = (cy - o[1]) / d[:, 1]
            ok = t > 0.05
            hx, hz = o[0] + t * d[:, 0], o[2] + t * d[:, 2]
            ok &= np.abs(_ceiling_at(apt, hx, hz) - cy) < 1e-6
            tmin = np.where(ok & (t < tmin), t, tmin)
        for f in faces:
            ax = f["axis"]
            t = (f["c"] - o[ax * 2]) / d[:, ax * 2]               # axis 0 -> x (idx0), axis 1 -> z (idx2)
            ok = t > 0.05
            tang = 2 if ax == 0 else 0
            ht = o[tang] + t * d[:, tang]; hy = o[1] + t * d[:, 1]
            ok &= (ht >= f["a0"]) & (ht <= f["a1"]) & (hy >= f["h0"]) & (hy <= f["h1"])
            tmin = np.where(ok & (t < tmin), t, tmin)
    z = tmin * d_cam[:, 2]                                 # d_cam z == 1 -> depth along optical axis == t
    z = z.reshape(Hh, W)
    z[~np.isfinite(z)] = 0
    rng = rng or np.random.default_rng(0)
    z = z + rng.normal(0, 1, z.shape) * (noise[0] + noise[1] * z) * (z > 0)
    z[(z < 0.25) | (z > 4.5)] = 0                         # sensor range
    return z


def simulate_capture(out_dir, apt=None, n_frames=420, seed=0, yaw_drift_deg_per_s=0.05, pos_drift_per_m=0.01,
                     floor_y=-1.45, rotation_deg=27.0, depth_noise=(0.003, 0.004), speed=0.5, fps=20):
    """Write a Record3D-format capture. World frame: Y up, floor at floor_y, plan rotated by rotation_deg about Y.
    Returns ground truth dict."""
    apt = apt or default_apartment()
    rng = np.random.default_rng(seed)
    faces = _faces(apt)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True); os.makedirs(os.path.join(out_dir, "confidence"), exist_ok=True)
    # trajectory: waypoints through rooms and doors; camera 1.4 m above the floor => y = 1.4 in plan frame
    way = [(2.0, 1.6), (0.9, 3.0), (1.05, 3.5), (1.0, 4.8), (2.2, 5.0), (1.0, 4.4), (1.05, 3.5), (2.0, 2.2),
           (3.7, 3.0), (3.7, 3.5), (3.7, 5.0), (3.7, 3.5), (3.7, 2.4), (4.3, 1.2), (5.6, 1.3), (4.3, 1.2), (3.0, 1.7), (2.0, 1.6)]
    pts = np.array(way)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]
    dur = cum[-1] / speed
    n = n_frames
    ts = np.linspace(0, dur, n)
    s = ts * speed
    xz = np.c_[np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])]
    fwd = np.gradient(xz, axis=0); heading = np.arctan2(fwd[:, 1], fwd[:, 0])
    # look around: sinusoidal yaw offset + pitch sweeps (so ceilings, floor and side walls are observed)
    yaw = heading + 1.2 * np.sin(2 * np.pi * ts / 6.0)
    pitch = 0.55 * np.sin(2 * np.pi * ts / 9.0 + 0.7)
    cy = np.radians(rotation_deg); Rw = Rot.from_euler("y", cy).as_matrix()     # plan->world rotation about Y
    K_rgb = np.array([[1600.0, 0, 955.5], [0, 1600.0, 717.8], [0, 0, 1]])
    Kd = K_rgb.copy(); Kd[0] *= 256 / 1920; Kd[1] *= 192 / 1440
    rows = []
    # drift model: yaw random-walk-ish ramp + position scale/rand walk
    yaw_drift = np.radians(yaw_drift_deg_per_s) * ts
    trav = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))]
    pos_walk = np.cumsum(rng.normal(0, 1, (n, 2)) * 0.002, axis=0)
    for i in range(n):
        # true camera pose in plan frame (x right? we use plan axes x, z horizontal, y up)
        cyaw, cp = yaw[i], pitch[i]
        fwd_w = np.array([np.cos(cyaw) * np.cos(cp), np.sin(cp), np.sin(cyaw) * np.cos(cp)])
        up_hint = np.array([0, 1.0, 0])
        zc = fwd_w / np.linalg.norm(fwd_w)
        xc = np.cross(zc, up_hint); xc /= np.linalg.norm(xc)          # camera x (right) ... y down = z x x
        yc = np.cross(zc, xc)
        R_plan = np.stack([xc, yc, zc], 1)
        # camera y points DOWN in CV frame; world up is +y => yc computed as zc x xc points down. Good.
        p_plan = np.array([xz[i, 0], 1.4, xz[i, 1]])
        depth = render_depth(apt, faces, R_plan, p_plan, Kd, rng=rng, noise=depth_noise)
        # --- observed (drifted) pose, expressed in the world frame ---
        R_obs = Rot.from_euler("y", yaw_drift[i]).as_matrix() @ Rw @ R_plan
        p_world = Rw @ p_plan
        p_obs = Rot.from_euler("y", yaw_drift[i]).as_matrix() @ p_world
        p_obs = p_obs + np.array([pos_walk[i, 0], 0, pos_walk[i, 1]]) + pos_drift_per_m * trav[i] * np.array([0.3, 0, -0.2])
        p_obs[1] = p_obs[1] + floor_y                                 # floor sits at floor_y; camera 1.4 above it
        q = Rot.from_matrix(R_obs).as_quat()
        cv2.imwrite(os.path.join(out_dir, "depth", f"{i:06d}.png"), np.round(depth * 1000).astype(np.uint16))
        cv2.imwrite(os.path.join(out_dir, "confidence", f"{i:06d}.png"), np.where(depth > 0, 2, 0).astype(np.uint8))
        rows.append((100.0 + ts[i], i, *p_obs, *q, K_rgb[0, 0], K_rgb[1, 1], K_rgb[0, 2], K_rgb[1, 2]))
    with open(os.path.join(out_dir, "odometry.csv"), "w") as f:
        f.write("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y\n")
        for r in rows:
            f.write(f"{r[0]:.6f}, {int(r[1]):06d}, " + ", ".join(f"{v:.8f}" for v in r[2:]) + ", , \n")
    np.savetxt(os.path.join(out_dir, "camera_matrix.csv"), K_rgb, delimiter=",")
    gt = ground_truth(apt)
    gt["sim"] = dict(seed=seed, rotation_deg=rotation_deg, yaw_drift_deg_per_s=yaw_drift_deg_per_s, pos_drift_per_m=pos_drift_per_m,
                     depth_noise=list(depth_noise), n_frames=n, floor_y=floor_y, synthetic=True)
    with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)
    return gt


def ground_truth(apt):
    rooms = {}
    for rid, r in apt["rooms"].items():
        x0, x1, z0, z1 = r["rect"]
        rooms[rid] = dict(walls=sorted([x1 - x0, z1 - z0, x1 - x0, z1 - z0]), dims=[x1 - x0, z1 - z0], ceiling_height=r["ceiling"],
                          floor_area=(x1 - x0) * (z1 - z0), openings=[])
    for d in apt["doors"]:
        w = d["u1"] - d["u0"]
        a, b = d["id"].split("-")
        for rid in (a, b):
            rooms[rid]["openings"].append(dict(id=d["id"], kind="door", width=w, height=d["h"]))
    return dict(rooms=rooms, doors=[dict(id=d["id"], width=d["u1"] - d["u0"], height=d["h"]) for d in apt["doors"]],
                footprint_area=sum(v["floor_area"] for v in rooms.values()))
