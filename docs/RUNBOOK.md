# RUNBOOK: from unzip to the final result

## 0. Setup (once)
```bash
unzip roomscan_project.zip && cd roomscan_project
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m unittest discover -s tests          # expect: OK (8 tests)
python -m roomscan selftest                   # expect: SELFTEST PASS
```
## 1. Put your sample data in place
Unzip each sample so each is a folder containing `rgb.mp4, depth/, confidence/, odometry.csv, imu.csv, camera_matrix.csv`:
```
data/single_room/            (from single_room.zip, the inner folder c00a170fe1)
data/single_scan_floor_only/
data/single_scan_with_ceiling/
```
(If your zip extracts to `c00a170fe1/...`, point the command at that inner folder.)
## 2. Run the pipeline on each
```bash
python -m roomscan run data/single_room -o out/single_room
python -m roomscan run data/single_scan_floor_only -o out/floor_only
python -m roomscan run data/single_scan_with_ceiling -o out/with_ceiling
```
Open `out/*/plan.png`. Floor-only scans report ceiling "NOT OBSERVED" on purpose. Compare with `examples/` (my run).
## 3. Drift ablation (needed for the drift gate)
```bash
python -m roomscan ablation data/single_scan_floor_only -o out/ablation   # footprint + wall sharpness: off / manhattan / full
```
## 4. Build YOUR benchmark (this is the part that decides your score)
Capture per docs/CAPTURE_PROTOCOL.md, one folder per scene under `bench/<scene>/` with a laser `ground_truth.json`:
```json
{"scene":"flat1","rooms":[{"label":"Living","walls":[4.20,3.40,4.20,3.40],"ceiling_height":2.55,"floor_area":14.28,
  "openings":[{"kind":"door","width":0.82}]}],"footprint_area":30.8,"adjacency":[["Living","Bed"]]}
```
```bash
python -m roomscan benchmark bench -o bench_out        # -> benchmark.json + benchmark_rows.csv (gates, errors, interval coverage)
python -m roomscan repeat bench/roomA_1 bench/roomA_2 -o repeat_out
python -m roomscan calibrate bench_out/benchmark_rows.csv -o intervals_calibrated.json   # then replace roomscan/data/intervals_default.json
python -m roomscan benchmark bench -o bench_out_after  # coverage should now be ~90 % (use a HELD-OUT split to report it honestly)
```
## 4b. Video tier (needs ML, optional)
```bash
pip install torch transformers pillow          # then first run downloads Depth-Anything-V2-Metric-Indoor (~1.3 GB) or set ROOMSCAN_DEPTH_MODEL to a local path
python -m roomscan run <folder with rgb.mp4 + odometry.csv> -o out --tier video
```
Untested by me. Compare against a LiDAR run of the same room before trusting it.

## 5. Head-to-head (10 %)
Capture the same 2 rooms with Polycam/magicplan (free tier), read their dimensions into `competitor.csv` (`room,kind,value`):
```bash
python -m roomscan headtohead out/x/plan.json competitor.csv bench/x/ground_truth.json -o h2h_out   # target: beat-or-tie >= 70 %
```
## 6. Fix loop (25 %)
`python -m roomscan benchmark` -> pick the worst gate -> fill docs/FIXLOOP.md -> **git commit** -> implement fix -> commit -> re-run benchmark at both commits. Keep both result folders.
## 7. Before submitting
`python scripts/compliance_check.py`, write the 6-page report from docs/REPORT_OUTLINE.md, commit as you go (history must be yours), run the cold rehearsal: someone else follows CAPTURE_PROTOCOL.md in a room you have never seen and you run `roomscan run ... --gt`.
