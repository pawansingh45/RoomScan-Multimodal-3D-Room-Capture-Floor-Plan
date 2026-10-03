"""Builds data/sim_bench/<scene>/ (synthetic Record3D captures + ground_truth.json in scoring format) and benchmarks them.
SYNTHETIC: validates algorithms and gives interval priors; it is NOT real-world accuracy evidence."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roomscan.sim.apartment import simulate_capture
from roomscan.eval.tools import benchmark

root = sys.argv[1] if len(sys.argv) > 1 else "data/sim_bench"
for i, (seed, rot) in enumerate([(11, 27.0), (12, 63.0), (13, 5.0)]):
    d = os.path.join(root, f"sim{i+1}")
    sim = simulate_capture(d, n_frames=500, seed=seed, rotation_deg=rot)
    gt = dict(scene=f"sim{i+1}", synthetic=True, rooms=[dict(label=k, walls=v["walls"], ceiling_height=v["ceiling_height"], floor_area=v["floor_area"],
              openings=[dict(kind="door", width=o["width"]) for o in v["openings"]]) for k, v in sim["rooms"].items()],
              footprint_area=sim["footprint_area"], adjacency=[x["id"].split("-") for x in sim["doors"]])
    json.dump(gt, open(os.path.join(d, "ground_truth.json"), "w"), indent=2)
rep = benchmark(root, os.path.join(root, "_results"))
for sc, r in rep.items():
    s = r["summary"]
    print(sc, "ceiling max|err| %.1f cm" % (100 * s["ceiling_height"]["max_abs"]), "| openings", s["opening_gate"], "| wall coverage %.2f" % s["wall_length"]["coverage"])
