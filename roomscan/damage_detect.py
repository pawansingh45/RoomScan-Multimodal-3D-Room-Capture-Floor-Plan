"""Image-based damage detection projected onto surfaces with metric extent.

Per keyframe (RGB at 480x360): anomaly maps vs the LOCAL surface appearance
   stain  : low-frequency blotch, darker + yellower/browner than its surroundings (Lab), blob >= ~6 cm
   mold   : dense dark speckle (black-hat) clusters
   crack  : thin elongated dark ridges (oriented black-hat, high aspect ratio)
Each anomalous pixel is lifted to 3D with the (pose-graph-corrected) LiDAR depth and assigned to a wall / ceiling surface of a
room. Evidence is voted per 2 cm surface cell across frames: damage = cells flagged in >= 50 % of the frames that saw them
(and >= 2 frames), so single-frame glare / motion blur cannot create damage. Floors are excluded (grout lines look like cracks).
Note: Classical morphological detectors are tuned for diffuse drywall surfaces.
Writes damage_detections.json consumed by damage.run_damage for automated scope line items.
"""
import os, json, logging
import numpy as np
import cv2
from scipy import ndimage as ndi

from .depth.stereo import read_frames
from .io.record3d import backproject
from .geom import rot_y
from .layout.rooms import to_manhattan

log = logging.getLogger("roomscan")


def anomaly_maps(bgr):
    """bgr (H,W,3) -> dict of float maps in [0,1]: stain, mold, crack."""
    img = cv2.GaussianBlur(bgr, (3, 3), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = cv2.GaussianBlur(lab, (0, 0), 24)                       # local surface appearance (low frequency)
    d = lab - bg
    dL, da, db = d[..., 0], d[..., 1], d[..., 2]
    # water stain: darker AND yellower/browner than the surroundings, smooth blob
    s = np.clip((-dL - 6) / 25.0, 0, 1) * np.clip((db + da * 0.5 - 4) / 14.0, 0, 1)
    s = cv2.GaussianBlur(s, (0, 0), 3)
    s = np.clip((s - 0.12) / 0.4, 0, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)
    speck = (bh > 28).astype(np.float32)
    mold = np.clip(cv2.GaussianBlur(speck, (0, 0), 6) * 6 - 0.8, 0, 1)
    best = np.zeros_like(gray, np.float32)
    for ang in range(0, 180, 22):
        k = np.zeros((19, 19), np.uint8)
        c = 9; dx, dy = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        cv2.line(k, (int(c - 9 * dx), int(c - 9 * dy)), (int(c + 9 * dx), int(c + 9 * dy)), 1, 1)
        best = np.maximum(best, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k).astype(np.float32))
    crack = (best > 30) & ~(bh > 28) | ((best > 38))
    lab_c, n = ndi.label(crack)
    ck = np.zeros_like(gray, np.float32)
    for i in range(1, n + 1):                                   # keep elongated components only
        ys, xs = np.nonzero(lab_c == i)
        if len(ys) < 25:
            continue
        ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys])) + 1e-6 * np.eye(2))
        if np.sqrt(ev[1] / max(ev[0], 1e-6)) > 5:
            ck[ys, xs] = 1
    return dict(water_stain=s, mold=mold, crack=ck)


def _surface_index(plan):
    """Walls of every room as (room, wall id, a, b, normal-sign) in the plan frame + room polygons/ceilings for lookup."""
    out = []
    for r in plan["rooms"]:
        H = r["ceiling_height"].get("value")
        for w in r["walls"]:
            a, b = np.array(w["start"]), np.array(w["end"]); L = np.linalg.norm(b - a)
            if L < 0.3:
                continue
            t = (b - a) / L
            out.append(dict(room=r["id"], wall=w["id"], a=a, t=t, L=L, n=np.array(w["outward"], float), H=H, floor_off=r.get("floor_height_offset_m", 0.0)))
    return out


def detect_damage(plan, extra, cfg, stride=4, cell=0.02, frame_hook=None, thr=0.35, min_votes=2, vote_frac=0.5):
    cap, kfs, pg, phi, floor_y = extra["cap"], extra["kfs"], extra["pg"], extra["phi"], extra["floor_y"]
    surfs = _surface_index(plan)
    if not surfs or cap.video_path() is None:
        return []
    sel = list(range(0, len(kfs), stride))
    frames = read_frames(cap.video_path(), {kfs[j].cap_idx for j in sel}, 0.25, cap.n, cap.t)
    acc = {}                                               # wall id -> dict(obs, hit[class]) over (iu, ih) cells
    for j in sel:
        k = kfs[j]; i = k.cap_idx
        g = frames.get(i)
        if g is None:
            continue
    # frames were decoded in gray for speed in read_frames; damage needs colour -> decode colour separately (below)
    frames = _read_colour(cap.video_path(), {kfs[j].cap_idx for j in sel}, cap)
    for j in sel:
        k = kfs[j]; i = k.cap_idx
        bgr = frames.get(i)
        d = cap.depth_m(i)
        if bgr is None or d is None:
            continue
        if frame_hook is not None:
            bgr = frame_hook(i, bgr, j)
        maps = anomaly_maps(bgr)
        pc, (uu, vv), m = backproject(d, cap.K_depth(i), cap.conf(i), cfg.depth_min_m, cfg.depth_max_m, cfg.min_confidence, cfg.edge_rel_thresh)
        if len(pc) == 0:
            continue
        Rw = rot_y(pg["psi"][j] - k.psi0) @ k.R0; p = pg["pos"][j]
        Pw = pc @ Rw.T + p
        q = to_manhattan(Pw[:, [0, 2]], phi); h = Pw[:, 1] - floor_y
        sx, sy = bgr.shape[1] / cap.depth_size[0], bgr.shape[0] / cap.depth_size[1]
        ru = np.clip(((uu + 0.5) * sx).astype(int), 0, bgr.shape[1] - 1); rv = np.clip(((vv + 0.5) * sy).astype(int), 0, bgr.shape[0] - 1)
        for sf in surfs:
            rel = q - sf["a"]; u = rel @ sf["t"]; dn = rel @ sf["n"]
            on = (np.abs(dn) < 0.07) & (u > 0.0) & (u < sf["L"]) & (h > 0.3) & (h < (sf["H"] or 2.6) + 0.05)
            if not on.any():
                continue
            iu = (u[on] / cell).astype(int); ih = (h[on] / cell).astype(int)
            A = acc.setdefault(sf["wall"], dict(room=sf["room"], L=sf["L"], obs={}, hit={}))
            keys = iu * 1000 + ih
            uk = np.unique(keys)
            for kk in uk:
                A["obs"][kk] = A["obs"].get(kk, 0) + 1                  # cell seen in this frame (once)
            for cls, mp in maps.items():
                flag = mp[rv[on], ru[on]] > thr
                for kk in np.unique(keys[flag]):
                    A["hit"].setdefault(cls, {}); A["hit"][cls][kk] = A["hit"][cls].get(kk, 0) + 1
    dets, n = [], 0
    for wid, A in acc.items():
        for cls, hits in A["hit"].items():
            cells = [kk for kk, c in hits.items() if A["obs"].get(kk, 0) >= min_votes and c / A["obs"][kk] >= vote_frac]
            if not cells:
                continue
            nu = max(kk // 1000 for kk in cells) + 3; nh = max(kk % 1000 for kk in cells) + 3
            img = np.zeros((nu, nh), bool)
            for kk in cells:
                img[kk // 1000, kk % 1000] = True
            img = ndi.binary_closing(img, iterations=2); lab, nn = ndi.label(img)
            for c in range(1, nn + 1):
                m_ = lab == c; area = m_.sum() * cell * cell
                min_area = {"water_stain": 0.02, "mold": 0.01, "crack": 0.0}[cls]
                if area < min_area or (cls == "crack" and m_.sum() < 8):
                    continue
                ii, hh = np.nonzero(m_)
                conf = float(np.mean([hits[ii_ * 1000 + hh_] / A["obs"][ii_ * 1000 + hh_] for ii_, hh_ in zip(ii, hh) if (ii_ * 1000 + hh_) in hits] or [0.5]))
                n += 1
                dets.append(dict(room=A["room"], surface=wid, **{"class": cls}, area_m2=round(float(area), 4),
                                 length_m=round(float(max(np.ptp(ii), np.ptp(hh)) * cell), 3) if cls == "crack" else None,
                                 height_range_m=[round(float(hh.min() * cell), 2), round(float(hh.max() * cell), 2)],
                                 u_range_m=[round(float(ii.min() * cell), 2), round(float(ii.max() * cell), 2)],
                                 confidence=round(min(0.95, conf), 2), method="classical_color_texture_votes"))
    return dets


def _read_colour(video_path, wanted, cap, scale=0.25):
    v = cv2.VideoCapture(video_path)
    nfr = int(v.get(cv2.CAP_PROP_FRAME_COUNT)); wanted = sorted(set(wanted)); last = wanted[-1] if wanted else -1
    map_ = {w: w for w in wanted}
    if nfr != cap.n:
        dur = nfr / max(v.get(cv2.CAP_PROP_FPS), 1e-6); vt = np.linspace(0, dur, nfr)
        map_ = {w: int(np.argmin(np.abs(vt - cap.t[w]))) for w in wanted}
    inv = {}
    for w, f in map_.items():
        inv.setdefault(f, []).append(w)
    out, idx = {}, 0
    while idx <= max(inv):
        if not v.grab():
            break
        if idx in inv:
            ok, fr = v.retrieve()
            if ok:
                im = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                for w in inv[idx]:
                    out[w] = im
        idx += 1
    v.release()
    return out
