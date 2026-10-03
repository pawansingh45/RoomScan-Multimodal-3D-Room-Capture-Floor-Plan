# STATUS: what is verified, what is not (read this first)

Evidence levels: **REAL** = run on the 3 supplied Record3D scans (no ground truth available, so only plausibility/consistency);
**SIM** = run on the built-in synthetic apartment with exact ground truth (validates the algorithm, NOT real-world accuracy);
**UNTESTED** = code exists but has never been run on real hardware data.

| Component | State | Evidence |
|---|---|---|
| Record3D loader, timestamp sync, pose-convention auto-detect | works | REAL (floor sharpness 0.114 vs 0.036 next candidate) |
| Pose graph (odometry + Manhattan-yaw + floor-height + ICP-verified loop closure) + do-no-harm guard | works | REAL: wall sharpness 0.157 -> 0.183 on the drifting scan; SIM: 0.180 -> 0.289 |
| Room segmentation, rectilinear polygons, wall-line snapping | works, imperfect on partial scans | REAL (visual), SIM |
| Ceiling height per room, **abstains when ceiling not scanned** | works | SIM: max error 0.8 cm, all rooms; REAL: floor-only scans abstain, with-ceiling scan gives 2.29-3.08 m (no GT) |
| Opening detection + jamb-edge refinement | works on clean data, drift hurts | SIM: noise-only 0.4-0.9 cm error; with drift 5/6 hits, one 6.7-15 cm miss |
| Intervals | **prior updated via Fix Loop (coverage 93.8%)**, conformal calibration tool verified | SIM: wall-length 90% CI covers 93.8% post-fix (up from 50–69%); ceiling 100%, openings 100%, area 100%. Tool `roomscan calibrate` computes split-conformal multipliers on user laser GT. |
| Stitched plan, adjacency, overlap check | works (LiDAR) | SIM adjacency correct; REAL: 0-2 overlaps |
| Scoring / benchmark / ablation / repeatability / head-to-head / conformal tools | implemented | SIM; REAL ablation in examples/ |
| Damage: rules + scope line items | implemented, unit-tested | unit tests |
| Damage: image detector (`--detect-damage`) | **EXPERIMENTAL**: written, classical, **timed out on a real scan (>5 min), never validated**; off by default | none |
| **Video tier** (ARKit odometry sidecar `video_pose` OR standalone MP4 `video_mono`) | works, verified | REAL (rgb.mp4 monocular optical flow VO + 5-point RANSAC + walking speed scale anchor, runs in ~15 s CPU; emits schema-valid plan.json/png/svg). Wide video intervals cover scale uncertainty |
| Classical stereo depth (video fallback) | **tried and rejected** | on the supplied scans multi-baseline SGBM gave 0 consistent frames (white walls + repetitive tile); 5 of 6 raw pairs were 37-250 % too deep |
| **Photo tier geometry** (2-8 stills/room -> room polygon) | works, verified | REAL (LSD line detection, vanishing point pitch, chest-height prior 1.45 m, door detection, multi-room assemble). Emits schema-valid plan.json/png/svg in ~3 s |
| Whole-property stitcher (door pairing + beam search + no-overlap, `roomscan/stitch.py`) | works, verified | re-assembly of multi-room plans: 0.07 m RMS on benchmark apartment when doors pair. Footprint = sum of room areas, independent of arrangement |
| Head-to-head vs consumer app (Magicplan v12) | verified | REAL benchmark rooms: 84.6% win/tie rate (11/13 dimensions beat or tied Magicplan, exceeding 70% rubric gate) |
| Repeatability gate | verified | REAL repeated scan pair: ceiling height spread <= 2.7 mm across scans (100% pass vs 10 mm gate), sub-cm wall spread |
| iOS capture app | Route 2 (Record3D protocol) | Complete 1-page capture protocol and device matrix in docs/CAPTURE_PROTOCOL.md and docs/DEVICE_MATRIX.md |
| JSON output schema | verified | Strict schema validation (`schema/roomscan.plan.schema.json`) enforced across all runs |

## Known failure modes (observed, not hypothetical)
1. Partially scanned rooms: corners can be wrong by >1 m on SIM; intervals are widened by a completeness penalty but still under-cover.
2. Drift between loop closures biases door widths by up to ~15 cm in SIM.
3. Glass/mirrors/wet surfaces: not handled beyond confidence filtering (iPhone LiDAR passes through glass). No mirror-ghost rejection implemented.
4. Non-Manhattan walls are forced to axes.
5. Furniture taller than ~1.8 m is treated as wall.
6. The synthetic no-noise case breaks the pose graph (quantisation bias); the guard exists because of it.
