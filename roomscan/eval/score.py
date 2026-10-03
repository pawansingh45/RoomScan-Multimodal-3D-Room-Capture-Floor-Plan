"""Score a plan.json against laser/tape ground truth.

Ground-truth file (JSON):
{ "scene": "flat1", "rooms": [ {"label": "Living", "walls": [4.20, 3.40, 4.20, 3.40], "ceiling_height": 2.55,
     "floor_area": 14.28, "openings": [{"kind": "door", "width": 0.82}] } ], "footprint_area": 30.8 }
Rooms/walls/openings are matched by Hungarian assignment on measured values (no coordinates needed), which is also
how the live walk-in score works: the laser values are typed in per room, not geo-referenced.
"""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

from .gates import GATES


def _covered(iv, gt):
    return iv is not None and iv.get("lo") is not None and iv["lo"] <= gt <= iv["hi"]


def _match_rooms(plan_rooms, gt_rooms):
    cost = np.zeros((len(gt_rooms), len(plan_rooms)))
    for i, g in enumerate(gt_rooms):
        gw = np.sort(np.array(g["walls"], float))[::-1]
        for j, p in enumerate(plan_rooms):
            pw = np.sort(np.array([w["length"]["value"] for w in p["walls"]]))[::-1][:len(gw)]
            pw = np.pad(pw, (0, len(gw) - len(pw)))
            ga = g.get("floor_area", np.nan)
            ca = abs(p["floor_area"]["value"] - ga) / ga if np.isfinite(ga) else 0
            cost[i, j] = np.abs(gw - pw).sum() / gw.sum() + ca
    r, c = linear_sum_assignment(cost) if cost.size else ([], [])
    return {int(i): int(j) for i, j in zip(r, c) if cost[i, j] < 1.0}


def score_plan(plan, gt, tier=None):
    tier = tier or plan["capture"]["tier"]
    rows, rooms = [], plan["rooms"]
    mapping = _match_rooms(rooms, gt["rooms"])
    wl_gate = GATES["wall_length"].get(tier, GATES["wall_length"]["lidar"])
    n_open_gt = n_open_hit = n_phantom = 0
    for gi, g in enumerate(gt["rooms"]):
        label = g.get("label", f"gt{gi}")
        if gi not in mapping:
            rows.append(dict(room=label, kind="room_missing", gt=None, pred=None, err=None, covered=False, passed=False))
            n_open_gt += len(g.get("openings", []))
            continue
        p = rooms[mapping[gi]]
        # walls
        gw = list(map(float, g["walls"]))
        pw = [w for w in p["walls"]]
        C = np.array([[abs(w["length"]["value"] - x) for w in pw] for x in gw])
        r, c = linear_sum_assignment(C)
        for i, j in zip(r, c):
            x, w = gw[i], pw[j]
            e = w["length"]["value"] - x
            tol = max(wl_gate.get("tol_m", 0), wl_gate.get("tol_rel", 0) * x)
            rows.append(dict(room=label, kind="wall_length", gt=x, pred=w["length"]["value"], err=e,
                             covered=_covered(w["length"], x), passed=abs(e) <= tol, observed=w["observed"], tol=tol,
                             width_ci=w["length"]["hi"] - w["length"]["lo"]))
        # ceiling
        if g.get("ceiling_height") is not None:
            ch = p["ceiling_height"]
            if ch.get("value") is None:
                rows.append(dict(room=label, kind="ceiling_height", gt=g["ceiling_height"], pred=None, err=None, covered=False,
                                 passed=False, abstained=True))
            else:
                e = ch["value"] - g["ceiling_height"]
                rows.append(dict(room=label, kind="ceiling_height", gt=g["ceiling_height"], pred=ch["value"], err=e,
                                 covered=_covered(ch, g["ceiling_height"]), passed=abs(e) <= GATES["ceiling_height"]["tol_m"],
                                 width_ci=ch["hi"] - ch["lo"]))
        # area
        if g.get("floor_area") is not None:
            fa = p["floor_area"]; e = fa["value"] - g["floor_area"]
            rows.append(dict(room=label, kind="floor_area", gt=g["floor_area"], pred=fa["value"], err=e, rel_err=e / g["floor_area"],
                             covered=_covered(fa, g["floor_area"]), passed=abs(e / g["floor_area"]) <= 0.03, width_ci=fa["hi"] - fa["lo"]))
        # openings (detection is scored)
        go = [o for o in g.get("openings", [])]
        po = p["openings"]
        n_open_gt += len(go)
        if go and po:
            C = np.array([[abs(o["width"]["value"] - x["width"]) if o["kind"] == x.get("kind", o["kind"]) or True else 9 for o in po] for x in go])
            r, c = linear_sum_assignment(C)
            used = set()
            for i, j in zip(r, c):
                if C[i, j] > 0.35:       # too far apart to be the same opening
                    continue
                used.add(j)
                e = po[j]["width"]["value"] - go[i]["width"]
                ok = abs(e) <= GATES["opening_width"]["tol_m"]
                n_open_hit += ok
                rows.append(dict(room=label, kind="opening_width", gt=go[i]["width"], pred=po[j]["width"]["value"], err=e,
                                 covered=_covered(po[j]["width"], go[i]["width"]), passed=ok, width_ci=po[j]["width"]["hi"] - po[j]["width"]["lo"]))
            matched_gt = {i for i, j in zip(r, c) if C[i, j] <= 0.35}
            for i in range(len(go)):
                if i not in matched_gt:
                    rows.append(dict(room=label, kind="opening_missed", gt=go[i]["width"], pred=None, err=None, covered=False, passed=False))
            for j in range(len(po)):
                if j not in used:
                    n_phantom += 1
                    rows.append(dict(room=label, kind="opening_phantom", gt=None, pred=po[j]["width"]["value"], err=None, covered=False, passed=False))
        else:
            for x in go:
                rows.append(dict(room=label, kind="opening_missed", gt=x["width"], pred=None, err=None, covered=False, passed=False))
            for o in po:
                n_phantom += 1
                rows.append(dict(room=label, kind="opening_phantom", gt=None, pred=o["width"]["value"], err=None, covered=False, passed=False))
    # footprint
    if gt.get("footprint_area") is not None and plan["stitched_plan"]["footprint_area"]["value"] is not None:
        fa = plan["stitched_plan"]["footprint_area"]; e = fa["value"] - gt["footprint_area"]
        rows.append(dict(room="*", kind="footprint", gt=gt["footprint_area"], pred=fa["value"], err=e, rel_err=e / gt["footprint_area"],
                         covered=_covered(fa, gt["footprint_area"]), passed=abs(e / gt["footprint_area"]) <= GATES["photo_stitch"]["footprint_rel"],
                         width_ci=fa["hi"] - fa["lo"]))
    # adjacency
    if gt.get("adjacency") is not None:
        gtadj = {tuple(sorted(a)) for a in gt["adjacency"]}
        inv = {gi: rooms[pj]["id"] for gi, pj in mapping.items()}
        lab_to_id = {gt["rooms"][gi].get("label", f"gt{gi}"): rid for gi, rid in inv.items()}
        pred = {tuple(sorted(c["rooms"])) for c in plan["stitched_plan"]["connections"]}
        want = {tuple(sorted((lab_to_id.get(a), lab_to_id.get(b)))) for a, b in gtadj}
        rows.append(dict(room="*", kind="adjacency", gt=len(want), pred=len(pred & want), err=len(pred - want),
                         covered=True, passed=(pred == want), extra=len(pred - want), missing=len(want - pred)))
    rows.append(dict(room="*", kind="overlaps", gt=0, pred=len(plan["stitched_plan"].get("overlaps", [])), err=None, covered=True,
                     passed=plan["stitched_plan"].get("overlap_free", True)))
    summ = summarise(rows, tier, n_open_gt, n_open_hit, n_phantom)
    return dict(rows=rows, summary=summ, mapping={gt["rooms"][i].get("label", i): rooms[j]["id"] for i, j in mapping.items()})


def summarise(rows, tier, n_open_gt=0, n_open_hit=0, n_phantom=0):
    out = {}
    for kind in ("wall_length", "ceiling_height", "floor_area", "opening_width", "footprint"):
        rr = [r for r in rows if r["kind"] == kind and r["err"] is not None]
        if rr:
            e = np.array([r["err"] for r in rr])
            out[kind] = dict(n=len(rr), mae=float(np.abs(e).mean()), median_abs=float(np.median(np.abs(e))), max_abs=float(np.abs(e).max()),
                             bias=float(e.mean()), spread_sd=float(e.std()), pass_rate=float(np.mean([r["passed"] for r in rr])),
                             coverage=float(np.mean([r["covered"] for r in rr])))
    denom = n_open_gt + n_phantom
    out["opening_gate"] = dict(n_gt=n_open_gt, hits=int(n_open_hit), phantom=int(n_phantom), missed=int(sum(r["kind"] == "opening_missed" for r in rows)),
                               fraction=(n_open_hit / denom) if denom else None,
                               passed=(n_open_hit / denom >= GATES["opening_width"]["min_fraction"]) if denom else None)
    cabs = [r for r in rows if r["kind"] == "ceiling_height"]
    if cabs:
        out["ceiling_gate"] = dict(n=len(cabs), abstained=int(sum(1 for r in cabs if r.get("abstained"))),
                                   passed_rooms=int(sum(r["passed"] for r in cabs)),
                                   verdict=("repeatable-but-biased" if (len([r for r in cabs if r["err"] is not None]) > 1 and
                                            abs(np.mean([r["err"] for r in cabs if r["err"] is not None])) > 2 * max(np.std([r["err"] for r in cabs if r["err"] is not None]), 1e-3))
                                            else "unbiased/noisy-or-n<2"))
    return out


def load_gt(path):
    with open(path) as f:
        return json.load(f)
