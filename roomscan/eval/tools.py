"""Evaluation tools: benchmark, drift ablation, repeatability, conformal calibration, head-to-head, fix-loop check."""
import os, json, csv, copy, subprocess, sys
import numpy as np

from ..config import Config
from ..tiers.lidar import run_lidar
from .score import score_plan, load_gt
from .gates import GATES


def _plan(path, cfg, mode=None):
    plan, _ = run_lidar(path, cfg, drift_mode=mode)
    return plan


def benchmark(scenes_dir, out_dir, cfg=None):
    """scenes_dir/<scene>/ contains a capture AND ground_truth.json (see eval/score.py for the format)."""
    cfg = cfg or Config(); os.makedirs(out_dir, exist_ok=True)
    allrows, report = [], {}
    for sc in sorted(os.listdir(scenes_dir)):
        d = os.path.join(scenes_dir, sc)
        gtp = os.path.join(d, "ground_truth.json")
        if not os.path.isdir(d) or not os.path.exists(gtp):
            continue
        plan = _plan(d, cfg)
        r = score_plan(plan, load_gt(gtp))
        for x in r["rows"]:
            allrows.append(dict(scene=sc, **x))
        report[sc] = dict(summary=r["summary"], timing_s=plan["diagnostics"]["total_s"])
    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    if allrows:
        keys = sorted({k for r in allrows for k in r})
        with open(os.path.join(out_dir, "benchmark_rows.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(allrows)
    return report


def ablation(path, out_dir, cfg=None):
    """Drift ablation: stitched footprint with pose-graph correction OFF / Manhattan-only / FULL (+ GT-free sharpness)."""
    cfg = cfg or Config(); os.makedirs(out_dir, exist_ok=True)
    res = {}
    for mode in ("off", "manhattan", "full"):
        p = _plan(path, cfg, mode)
        res[mode] = dict(footprint_m2=p["stitched_plan"]["footprint_area"]["value"], n_rooms=p["stitched_plan"]["n_rooms"],
                         overlaps=len(p["stitched_plan"].get("overlaps", [])), wall_sharpness=p["diagnostics"]["wall_sharpness"],
                         drift=p["diagnostics"]["drift"], room_areas=[r["floor_area"]["value"] for r in p["rooms"]])
    with open(os.path.join(out_dir, "drift_ablation.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    return res


def repeatability(path_a, path_b, out_dir, cfg=None):
    """Two captures of the same room at the same tier: per-wall |diff| vs 1 cm / 0.5 % gate, ceiling spread vs 1 cm."""
    cfg = cfg or Config(); os.makedirs(out_dir, exist_ok=True)
    A, B = _plan(path_a, cfg), _plan(path_b, cfg)
    from scipy.optimize import linear_sum_assignment
    ra = sorted(A["rooms"], key=lambda r: -r["floor_area"]["value"]); rb = sorted(B["rooms"], key=lambda r: -r["floor_area"]["value"])
    rows = []
    for a, b in zip(ra, rb):
        la = np.array([w["length"]["value"] for w in a["walls"]]); lb = np.array([w["length"]["value"] for w in b["walls"]])
        C = np.abs(la[:, None] - lb[None, :]); i, j = linear_sum_assignment(C)
        for x, y in zip(i, j):
            d = abs(la[x] - lb[y]); tol = max(GATES["repeatability"]["tol_m"], GATES["repeatability"]["tol_rel"] * la[x])
            rows.append(dict(room=a["id"], kind="wall", a=la[x], b=lb[y], diff=d, tol=tol, passed=bool(d <= tol)))
        ca, cb = a["ceiling_height"]["value"], b["ceiling_height"]["value"]
        if ca and cb:
            rows.append(dict(room=a["id"], kind="ceiling", a=ca, b=cb, diff=abs(ca - cb), tol=GATES["ceiling_height"]["spread_m"], passed=bool(abs(ca - cb) <= GATES["ceiling_height"]["spread_m"])))
    out = dict(rows=rows, pass_rate=float(np.mean([r["passed"] for r in rows])) if rows else None)
    with open(os.path.join(out_dir, "repeatability.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def calibrate(rows_csv, model_path_in, model_path_out):
    """Split-conformal calibration: q = (1-alpha)(1+1/n)-quantile of |err|/sigma per (tier, measure). Rows must carry
    err and the interval half-width (width_ci/2 = z*sigma*q_old). Run on a CALIBRATION split, report coverage on a HELD-OUT split."""
    from scipy.stats import norm
    with open(rows_csv) as f:
        rows = list(csv.DictReader(f))
    with open(model_path_in) as f:
        m = json.load(f)
    level = 0.90; z = norm.ppf(0.5 + level / 2)
    mapk = {"wall_length": "length", "ceiling_height": "ceiling", "opening_width": "opening", "floor_area": "area"}
    tier = "lidar"
    for kind, name in mapk.items():
        rr = [r for r in rows if r["kind"] == kind and r.get("err") not in ("", None) and r.get("width_ci") not in ("", None)]
        if len(rr) < 8:
            continue
        err = np.abs([float(r["err"]) for r in rr]); half = np.array([float(r["width_ci"]) / 2 for r in rr])
        score = err / np.maximum(half / z, 1e-6)                       # |err|/ (sigma*q_old)
        n = len(score); qhat = float(np.quantile(score, min(1.0, level * (1 + 1 / n))))
        m["q"][tier][name] = float(m["q"][tier][name] * qhat / z)
    m["calibrated"] = True; m["source"] = "split conformal on user benchmark"
    with open(model_path_out, "w") as f:
        json.dump(m, f, indent=2)
    return m


def head_to_head(our_plan_json, competitor_csv, gt_json, out_dir, tie_m=0.005):
    """competitor_csv columns: room,kind,value   (kind in wall_length|ceiling_height|opening_width; values in metres, read off the
    competitor's export). Compared against the SAME ground truth, dimension by dimension; 'tie' = |err_diff| < tie_m."""
    os.makedirs(out_dir, exist_ok=True)
    with open(our_plan_json) as f:
        plan = json.load(f)
    gt = load_gt(gt_json)
    ours = score_plan(plan, gt)["rows"]
    with open(competitor_csv) as f:
        theirs = list(csv.DictReader(f))
    # competitor values are matched to GT exactly as ours (Hungarian on value) per (room, kind)
    from scipy.optimize import linear_sum_assignment
    table, wins = [], 0
    for room in {r["room"] for r in theirs}:
        for kind in ("wall_length", "ceiling_height", "opening_width"):
            ours_k = [x for x in ours if x["room"] == room and x["kind"] == kind and x["gt"] is not None]
            g = [x["gt"] for x in ours_k]
            t = [float(r["value"]) for r in theirs if r["room"] == room and r["kind"] == kind]
            if not g or not t:
                continue
            C = np.abs(np.array(g)[:, None] - np.array(t)[None, :]); i, j = linear_sum_assignment(C)
            for a, b in zip(i, j):
                eo = abs(ours_k[a]["err"] or 0)
                et = abs(t[b] - g[a])
                res = "tie" if abs(eo - et) < tie_m else ("win" if eo < et else "loss")
                wins += res in ("win", "tie")
                table.append(dict(room=room, kind=kind, gt=g[a], ours_err=eo, theirs_err=et, result=res))
    out = dict(table=table, beat_or_tie_fraction=(wins / len(table)) if table else None, target=0.70, tie_definition=f"|err_ours-err_theirs|<{tie_m} m")
    with open(os.path.join(out_dir, "head_to_head.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def fixloop_check(declaration_md):
    """Fix-loop hygiene: the declaration must name gate, failing number, hypothesis+evidence, intended fix, predicted number."""
    with open(declaration_md) as f:
        txt = f.read().lower()
    need = ["worst gate", "failing number", "hypothesis", "evidence", "fix", "predicted"]
    missing = [k for k in need if k not in txt]
    return dict(ok=not missing, missing=missing)
