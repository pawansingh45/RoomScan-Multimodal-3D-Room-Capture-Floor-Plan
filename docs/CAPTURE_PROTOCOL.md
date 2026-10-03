# One-page capture protocol (Route 2: stock app, LiDAR tier)  -- for a non-engineer

**Device:** iPhone 12 Pro / 13 Pro / 14 Pro / 15 Pro or newer Pro (needs the LiDAR dot on the back).
**Install:** "Record3D" from the App Store. Settings: **Record to file (RGBD, .r3d / video+depth export)**; export as the *"Shareable/External"* format that gives rgb.mp4 + depth/ + confidence/ + odometry.csv + imu.csv + camera_matrix.csv (the layout of the sample data). Verify the export folder looks like the samples BEFORE the day.

**Per property (10-20 min)**
1. Open every door you want in the plan. Turn lights ON. Close blinds if sun glares on glass. Cover or avoid mirrors.
2. Start recording at the entrance. Hold the phone upright, chest height, **walking slowly (about one step per second)**.
3. In every room: stand in the doorway, then walk the room's edge; **aim at the floor line, each wall top to bottom, then sweep the CEILING for ~5 s** (ceiling height needs it). Do not skip corners.
4. Pass through every doorway slowly, looking at both jambs. Finish by **walking back to the start** (closes the loop; it measurably reduces drift).
5. Avoid: fast turns, pointing at mirrors/glass/black glossy surfaces, people walking through, covering the LiDAR dot.
6. Stop recording. Export to a folder. Name it `<property>_<date>` and give us the folder (AirDrop / cable).

**Ground truth (benchmark only):** laser-measure each wall at ~1 m height (corner-to-corner of the finished surface), ceiling height at 3-4 points, every door/opening clear width jamb-to-jamb at ~1 m. Write `ground_truth.json` (format in roomscan/eval/score.py).

**Video / photo tiers (not yet supported by the code):** stock Camera app, iPhone 15+. Video: one slow walk-through, landscape-free, ending where it started. Photos: 2-8 per room, one folder per room, include each doorway from inside and outside, hold phone at chest height.

Run: `python -m roomscan run <folder> -o out` -> `out/plan.png`, `out/plan.json`.
