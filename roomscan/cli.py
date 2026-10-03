import argparse, json, logging, os, sys
from .config import Config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="roomscan", description="phone capture -> dimensioned, stitched floor plan with confidence intervals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ONE command per capture: roomscan run <capture_dir> -o <out_dir>")
    r.add_argument("capture"); r.add_argument("-o", "--out", default="out"); r.add_argument("--tier", default="auto", choices=["auto", "lidar", "video", "photo"])
    r.add_argument("--drift", default=None, choices=["off", "manhattan", "full"]); r.add_argument("--config"); r.add_argument("--gt", help="laser ground-truth json: score immediately")
    r.add_argument("--no-damage", action="store_true"); r.add_argument("--detect-damage", action="store_true", help="EXPERIMENTAL classical damage detector (slow, unverified)"); r.add_argument("-q", "--quiet", action="store_true")
    s = sub.add_parser("score", help="score a plan.json against laser ground truth"); s.add_argument("plan"); s.add_argument("gt")
    b = sub.add_parser("benchmark", help="run + score every scene dir (capture + ground_truth.json)"); b.add_argument("scenes"); b.add_argument("-o", "--out", default="bench_out")
    a = sub.add_parser("ablation", help="drift ablation off/manhattan/full"); a.add_argument("capture"); a.add_argument("-o", "--out", default="ablation_out")
    rp = sub.add_parser("repeat", help="repeatability of two captures of the same room"); rp.add_argument("a"); rp.add_argument("b"); rp.add_argument("-o", "--out", default="repeat_out")
    c = sub.add_parser("calibrate", help="split-conformal interval calibration from benchmark_rows.csv"); c.add_argument("rows_csv"); c.add_argument("-o", "--out", default="intervals_calibrated.json")
    h = sub.add_parser("headtohead", help="compare against a competitor export"); h.add_argument("plan"); h.add_argument("competitor_csv"); h.add_argument("gt"); h.add_argument("-o", "--out", default="h2h_out")
    sub.add_parser("selftest", help="synthetic end-to-end self test (no data needed)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if getattr(args, "quiet", False) else logging.INFO, format="%(message)s")
    if args.cmd == "run":
        from .pipeline import run
        cfg = Config.load(args.config)
        if args.detect_damage:
            cfg.damage_detect = True
        plan = run(args.capture, args.out, args.tier, cfg, args.drift, damage=not args.no_damage)
        print_summary(plan, args.out)
        if args.gt:
            from .eval.score import score_plan, load_gt
            res = score_plan(plan, load_gt(args.gt)); print_score(res)
            with open(os.path.join(args.out, "score.json"), "w") as f:
                json.dump(res, f, indent=2, default=float)
    elif args.cmd == "score":
        from .eval.score import score_plan, load_gt
        with open(args.plan) as pf, open(args.gt) as gf:
            print_score(score_plan(json.load(pf), load_gt(args.gt)))
    elif args.cmd == "benchmark":
        from .eval.tools import benchmark; print(json.dumps(benchmark(args.scenes, args.out), indent=1, default=float))
    elif args.cmd == "ablation":
        from .eval.tools import ablation; print(json.dumps({k: {x: v[x] for x in ("footprint_m2", "wall_sharpness", "overlaps")} for k, v in ablation(args.capture, args.out).items()}, indent=1, default=float))
    elif args.cmd == "repeat":
        from .eval.tools import repeatability; print(json.dumps(repeatability(args.a, args.b, args.out)["pass_rate"]))
    elif args.cmd == "calibrate":
        from .eval.tools import calibrate; from .uncertainty import _DEF; calibrate(args.rows_csv, _DEF, args.out); print("wrote", args.out)
    elif args.cmd == "headtohead":
        from .eval.tools import head_to_head; print(json.dumps(head_to_head(args.plan, args.competitor_csv, args.gt, args.out), indent=1, default=float))
    elif args.cmd == "selftest":
        return selftest()


def print_summary(plan, out):
    cap, sp = plan["capture"], plan["stitched_plan"]
    print(f"\n== {cap['tier']} | status {cap['status']} | rooms {sp['n_rooms']} | wall-clock {plan['diagnostics'].get('wall_clock_s')} s ==")
    for r in plan["rooms"]:
        ch = r["ceiling_height"]; fa = r["floor_area"]
        c = f"{ch['value']:.3f} [{ch['lo']:.3f},{ch['hi']:.3f}]" if ch.get("value") else "NOT OBSERVED (abstained)"
        print(f" {r['id']:4s} {r['kind']:9s} area {fa['value']:6.2f} [{fa['lo']:.2f},{fa['hi']:.2f}] m2 | ceiling {c} | walls {len(r['walls'])} | openings {[round(o['width']['value'],2) for o in r['openings']]} | completeness {r['completeness']}")
    print(f" connections: {sp['adjacency']} | overlaps: {len(sp['overlaps'])} | footprint {sp['footprint_area']['value']:.1f} m2")
    print(f" outputs: {out}/plan.json  {out}/plan.png  {out}/plan.svg")
    if not plan["diagnostics"].get("intervals_calibrated", False):
        print(" NOTE: intervals are engineering priors (uncalibrated) until you run `roomscan calibrate` on laser ground truth.")


def print_score(res):
    print("\n== score vs ground truth ==")
    for k, v in res["summary"].items():
        print(f" {k}: {json.dumps(v, default=float)}")


def selftest():
    import tempfile
    from .sim.apartment import simulate_capture
    from .pipeline import run
    from .eval.score import score_plan
    d = tempfile.mkdtemp()
    sim = simulate_capture(os.path.join(d, "sim"), n_frames=400, seed=3)
    plan = run(os.path.join(d, "sim"), os.path.join(d, "out"), damage=False)
    gt = dict(rooms=[dict(label=k, walls=v["walls"], ceiling_height=v["ceiling_height"], floor_area=v["floor_area"], openings=[dict(kind="door", width=o["width"]) for o in v["openings"]]) for k, v in sim["rooms"].items()], footprint_area=sim["footprint_area"])
    res = score_plan(plan, gt); print_score(res)
    ok = plan["capture"]["status"] == "ok" and plan["stitched_plan"]["n_rooms"] >= 3
    print("SELFTEST", "PASS" if ok else "FAIL"); return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
