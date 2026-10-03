import unittest, sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage as ndi


class T(unittest.TestCase):
    def test_icp_recovers_transform(self):
        from roomscan.lidar.posegraph import icp_se2, R2
        rng = np.random.default_rng(0)
        w = np.vstack([np.c_[np.linspace(0, 5, 500), np.zeros(500)], np.c_[np.zeros(300), np.linspace(0, 3, 300)], np.c_[np.full(200, 5.0), np.linspace(0, 2, 200)]])
        w += rng.normal(0, .003, w.shape)
        th, tau = 0.05, np.array([.12, -.08])
        r = icp_se2(w, (w - tau) @ R2(th))
        self.assertAlmostEqual(r["theta"], th, 3); self.assertLess(np.abs(r["tau"] - tau).max(), 5e-3)

    def test_boundary_tracer_is_rectilinear(self):
        from roomscan.layout.rooms import trace_boundary
        rng = np.random.default_rng(1)
        for _ in range(60):
            A = ndi.binary_fill_holes(ndi.gaussian_filter(rng.random((50, 50)), 3) > .5)
            p = trace_boundary(A)
            if p is None: continue
            d = np.abs(np.diff(np.vstack([p, p[:1]]), axis=0))
            self.assertFalse(((d[:, 0] > 0) & (d[:, 1] > 0)).any())

    def test_simplifier_keeps_rectilinear(self):
        from roomscan.layout.rooms import _rectilinear_simplify
        q = np.array([[0, 0], [0, 1], [.05, 1], [.05, 1.5], [0, 1.5], [0, 2], [3, 2], [3, 0]], float)
        out = _rectilinear_simplify(q, .15)
        self.assertEqual(len(out), 4)

    def test_schema_validator_rejects_bad(self):
        from roomscan.schema import validate_plan
        self.assertTrue(validate_plan({"schema": "x"}))

    def test_interval_widens_with_thin_data(self):
        from roomscan.uncertainty import IntervalModel
        m = IntervalModel()
        w = lambda t, o: m.wall_length(4.0, t, o)["hi"] - m.wall_length(4.0, t, o)["lo"]
        self.assertLess(w("lidar", 2), w("lidar", 0)); self.assertLess(w("lidar", 2), w("photo", 2))

    def test_rules_name_the_rule_that_fired(self):
        from roomscan.damage import rule_engine
        room = dict(id="R1", ceiling_height=dict(value=2.4))
        d = [dict(id="R1.d1", surface_id="R1.w1", **{"class": "water_stain"}, confidence=.8, height_range_m=[2.0, 2.4], area_m2=.3)]
        f = rule_engine(room, d)
        self.assertTrue(any(x["rule"].startswith("R1_") for x in f))

    def test_pipeline_on_simulated_apartment(self):
        from roomscan.sim.apartment import simulate_capture
        from roomscan.pipeline import run
        from roomscan.eval.score import score_plan
        d = tempfile.mkdtemp()
        sim = simulate_capture(os.path.join(d, "s"), n_frames=300, seed=5)
        plan = run(os.path.join(d, "s"), os.path.join(d, "o"), damage=False, render=False)
        self.assertEqual(plan["capture"]["status"], "ok"); self.assertGreaterEqual(plan["stitched_plan"]["n_rooms"], 3)
        gt = dict(rooms=[dict(label=k, walls=v["walls"], ceiling_height=v["ceiling_height"], floor_area=v["floor_area"], openings=[]) for k, v in sim["rooms"].items()])
        s = score_plan(plan, gt)["summary"]["ceiling_height"]
        self.assertLess(s["max_abs"], 0.03)

    def test_stitcher_places_rooms_with_door_evidence(self):
        from roomscan.stitch import assemble, rooms_from_plan, placement_error, footprint_interval
        import numpy as np
        sq = lambda x0, y0, w, h: [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]
        mk = lambda rid, poly, doors: dict(id=rid, polygon=poly, openings=[dict(id=rid + ".o", wall_id=rid + ".w1", kind="door", start=d[0], end=d[1], width=dict(value=d[2])) for d in doors],
                                          walls=[dict(id=rid + ".w1", outward=d[3]) for d in doors[:1]])
        plan = dict(rooms=[mk("R1", sq(0, 0, 4, 3), [([1.0, 3.0], [1.9, 3.0], 0.9, [0, 1])]), mk("R2", sq(0, 3.15, 3, 2.5), [([1.0, 3.15], [1.9, 3.15], 0.9, [0, -1])])])
        rooms, truth = rooms_from_plan(plan, np.random.default_rng(1))
        L = assemble(rooms)
        self.assertEqual(L[0]["unplaced"], []); self.assertLess(placement_error(L[0], truth), 0.1)
        self.assertAlmostEqual(footprint_interval(L)["value"], 4 * 3 + 3 * 2.5, 2)

    def test_video_tier_refuses_without_provider_or_sidecar(self):
        from roomscan.tiers.video import run_video
        from roomscan.config import Config
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "clip.mp4"), "wb") as f:
            f.write(b"x")
        p = run_video(d, Config(), d)
        self.assertEqual(p["capture"]["status"], "insufficient"); self.assertEqual(p["rooms"], [])

    def test_failure_still_emits_valid_plan(self):
        from roomscan.pipeline import run
        from roomscan.schema import validate_plan
        d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "x", "depth"))
        with open(os.path.join(d, "x", "odometry.csv"), "w") as f:
            f.write("timestamp,frame\n")
        p = run(os.path.join(d, "x"), os.path.join(d, "o"), render=False)
        self.assertEqual(validate_plan(p), []); self.assertIn(p["capture"]["status"], ("failed", "insufficient"))


if __name__ == "__main__":
    unittest.main()
