"""Assemble the output plan (per-room + stitched whole-property) from rasters, regions and detections."""
import logging
import numpy as np
import cv2

from .layout.rooms import from_manhattan
from .layout.vertical import estimate_vertical
from .layout.openings import edge_frame, detect_openings_on_edge

log = logging.getLogger("roomscan")


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def rasterize_poly(poly, grid):
    img = np.zeros(grid.shape[::-1], np.uint8)    # (cols=j, rows=i) for cv2 (x=j,y=i) ... we store [i,j]
    pts = ((poly - grid.mn) / grid.res)
    pts_cv = np.round(np.c_[pts[:, 1], pts[:, 0]]).astype(np.int32)   # cv2 wants (x=col=j, y=row=i)
    m = np.zeros(grid.shape, np.uint8)
    cv2.fillPoly(m, [pts_cv], 1)
    return m.astype(bool)


def assemble_plan(rs, labels, rooms, tier, cfg, im, phi, meta=None):
    meta = meta or {}
    grid = rs.grid
    # stable room numbering: left-to-right then bottom-to-top
    order = sorted(range(len(rooms)), key=lambda i: (round(float(rooms[i]["poly_q"][:, 0].mean()), 1), float(rooms[i]["poly_q"][:, 1].mean())))
    out_rooms, label_to_id = [], {}
    for rank, i in enumerate(order, 1):
        r = rooms[i]
        poly = np.array(r["poly_q"]); lines = list(r["lines"])
        if poly_area(poly) < 0:                       # force counter-clockwise
            n = len(poly)
            poly = poly[::-1].copy()
            lines = [lines[(n - 2 - k) % n] for k in range(n)]
        label_to_id[r["label"]] = f"R{rank}"
        r["_poly"], r["_lines"], r["_id"] = poly, lines, f"R{rank}"
    for i in order:
        r = rooms[i]
        poly, lines, rid = r["_poly"], r["_lines"], r["_id"]
        area = poly_area(poly)
        v = estimate_vertical(rs, r["mask"], area, cfg)
        ceil_obs = v.get("ceiling_status") == "observed"
        H = (v["ceiling_y"] - v["floor_offset"]) if ceil_obs else None
        n = len(poly)
        obs = [bool(l[2]) for l in lines]
        walls, openings = [], []
        edge_sig = []
        Ls = [float(np.linalg.norm(poly[(k + 1) % n] - poly[k])) for k in range(n)]
        unobs_frac = sum(L_ for L_, o_ in zip(Ls, obs) if not o_) / max(sum(Ls), 1e-6)
        partial_extra = 2.0 * unobs_frac + 3.0 * unobs_frac ** 2   # Quadratic expansion for partially observed boundaries
        for k in range(n):
            a, b = poly[k], poly[(k + 1) % n]
            L = float(np.linalg.norm(b - a))
            ends_obs = int(obs[(k - 1) % n]) + int(obs[(k + 1) % n])
            spread = float(lines[k][4]) if np.isfinite(lines[k][4]) else 0.0
            wid = f"{rid}.w{k + 1}"
            fr = edge_frame(poly, k, r["mask"], grid)
            det = dict(openings=[], support=0.0)
            if L >= 0.5:
                det = detect_openings_on_edge(rs, fr, H or 2.4, cfg)
            support = det["support"]
            kind = "wall" if (obs[k] and support >= 0.25) else ("inferred" if not obs[k] else "partial")
            edge_sig.append((L, im.edge_offset_sigma(tier, obs[k])))
            wall = dict(id=wid, start=[float(a[0]), float(a[1])], end=[float(b[0]), float(b[1])],
                        length=im.wall_length(L, tier, ends_obs, spread, extra=partial_extra), observed=bool(obs[k]), kind=kind,
                        n_points=int(lines[k][3]), support=round(float(support), 3),
                        outward=[float(fr["sgn"]), 0.0] if fr["ax_c"] == 0 else [0.0, float(fr["sgn"])],
                        openings=[])
            for j, o in enumerate(det["openings"], 1):
                oid = f"{wid}.o{j}"
                pa = np.zeros(2); pb = np.zeros(2)
                pa[fr["ax_t"]], pb[fr["ax_t"]] = o["t0"], o["t1"]; pa[fr["ax_c"]] = pb[fr["ax_c"]] = fr["c"]
                mid = 0.5 * (pa + pb)
                probe = mid.copy(); probe[fr["ax_c"]] += fr["sgn"] * 0.35
                ij = grid.ij(probe[None])[0]
                lab = 0
                if 0 <= ij[0] < grid.shape[0] and 0 <= ij[1] < grid.shape[1]:
                    lab = int(labels[ij[0], ij[1]])
                leads = label_to_id.get(lab) if lab and lab != r["label"] else None
                confirmed = o["through"] >= 0.5
                height = None
                if o["head"] is not None:
                    height = im._iv(o["head"] - o["sill"], im.p["tiers"][im.tier_key(tier)]["open_abs"] * 1.5, tier, "opening", basis="head - sill")
                op = dict(id=oid, wall_id=wid, kind=o["kind"], width=im.opening_width(o["width"], tier, confirmed),
                          height=height, sill_m=round(o["sill"], 3), offset_from_wall_start=im._iv(o["u0"], 0.03, tier, "opening"),
                          start=[float(pa[0]), float(pa[1])], end=[float(pb[0]), float(pb[1])], leads_to=leads,
                          evidence=dict(see_through=round(o["through"], 2), confirmed=bool(confirmed)))
                openings.append(op); wall["openings"].append(oid)
            walls.append(wall)
        room = dict(id=rid, name=f"Room {rid[1:]}", polygon=[[float(x), float(y)] for x, y in poly],
                    walls=walls, openings=openings, completeness=round(1.0 - unobs_frac, 3),
                    floor_area=im.area(area, edge_sig, tier),
                    perimeter=float(sum(np.linalg.norm(poly[(k + 1) % n] - poly[k]) for k in range(n))),
                    bbox_m=[float(np.ptp(poly[:, 0])), float(np.ptp(poly[:, 1]))],
                    centroid=[float(poly[:, 0].mean()), float(poly[:, 1].mean())],
                    floor_height_offset_m=round(float(v.get("floor_offset", 0.0)), 4),
                    vertical=dict(floor_status=v.get("floor_status"), ceiling_status=v.get("ceiling_status"),
                                  ceiling_cover=round(float(v.get("ceiling_cover", 0.0)), 3)),
                    surfaces=[], damage=[], concealed_damage_flags=[], scope_items=[])
        if ceil_obs:
            room["ceiling_height"] = dict(im.ceiling(H, tier, v["ceiling_sigma"]), status="observed")
            if "ceiling_secondary_y" in v:
                room["ceiling_height"]["secondary_level_m"] = round(float(v["ceiling_secondary_y"] - v["floor_offset"]), 3)
        else:
            lb = v.get("wall_top_lower_bound")
            room["ceiling_height"] = dict(value=None, lo=None, hi=None, unit="m", level=im.level, status="not_observed",
                                          lower_bound=None if lb is None else round(float(lb), 3),
                                          basis="ceiling plane not scanned; only a lower bound from the highest wall return",
                                          calibrated=bool(im.p["calibrated"]))
        out_rooms.append(room)
    # ----- connections (stitching) -----
    conn = {}
    for r in out_rooms:
        for o in r["openings"]:
            if o["leads_to"]:
                key = tuple(sorted((r["id"], o["leads_to"])))
                conn.setdefault(key, []).append((r["id"], o))
    connections = []
    for (a, b), items in conn.items():
        sides = {rid for rid, _ in items}
        ws = [o["width"]["value"] for _, o in items]
        connections.append(dict(rooms=[a, b], openings=[o["id"] for _, o in items], width_m=float(np.mean(ws)),
                                confidence="both_sides" if len(sides) == 2 else "single_side"))
    deg = {r["id"]: 0 for r in out_rooms}
    for c in connections:
        for rid in c["rooms"]:
            deg[rid] += 1
    for r in out_rooms:
        mn, mx = sorted(r["bbox_m"])
        r["kind"] = "connector" if ((mn < 1.6 and mx / max(mn, 1e-6) > 2.5) or deg[r["id"]] >= 4) else "room"
        r["n_connections"] = deg[r["id"]]
    # ----- stitched plan checks: overlaps and footprint -----
    masks = {r["id"]: rasterize_poly(np.array(r["polygon"]), grid) for r in out_rooms}
    ids = list(masks)
    overlaps, slivers = [], []
    from scipy import ndimage as _ndi
    eroded = {k: _ndi.binary_erosion(m_, iterations=2) for k, m_ in masks.items()}     # 2 cells = 4 cm per side
    for i_ in range(len(ids)):
        for j_ in range(i_ + 1, len(ids)):
            raw = float((masks[ids[i_]] & masks[ids[j_]]).sum() * grid.res ** 2)
            real = float((eroded[ids[i_]] & eroded[ids[j_]]).sum() * grid.res ** 2)
            if real > 0.02:                       # survives 8 cm of erosion: a genuine overlap
                overlaps.append(dict(rooms=[ids[i_], ids[j_]], area_m2=round(real, 3)))
            elif raw > 0.02:                      # thinner than a wall: shared-wall snapping sliver, reported for transparency
                slivers.append(dict(rooms=[ids[i_], ids[j_]], area_m2=round(raw, 3)))
    union = np.zeros(grid.shape, bool)
    for m_ in masks.values():
        union |= m_
    foot = float(union.sum() * grid.res ** 2)
    all_edges = [(L, s) for r in out_rooms for L, s in [(w["length"]["value"], im.edge_offset_sigma(tier, w["observed"])) for w in r["walls"]]]
    stitched = dict(frame="plan (Manhattan-aligned; x right, y up)", footprint_area=im.area(foot, all_edges, tier),
                    n_rooms=len(out_rooms), connections=connections, overlaps=overlaps, overlap_free=len(overlaps) == 0, sliver_overlaps_thinner_than_wall=slivers,
                    bbox_m=[float(np.ptp(np.vstack([np.array(r["polygon"]) for r in out_rooms])[:, 0])),
                            float(np.ptp(np.vstack([np.array(r["polygon"]) for r in out_rooms])[:, 1]))] if out_rooms else [0, 0],
                    adjacency=sorted([c["rooms"] for c in connections]))
    return dict(schema="roomscan.plan/1.0", rooms=out_rooms, stitched_plan=stitched, plan_frame=dict(phi_deg=float(np.degrees(phi)), up="+Y (gravity)",
                note="plan x/y = ARKit world (x,z) rotated by phi so walls are axis-aligned"))
