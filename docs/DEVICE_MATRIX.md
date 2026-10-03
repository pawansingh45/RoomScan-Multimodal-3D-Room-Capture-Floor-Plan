# Device Matrix (Comprehensive & Verified)

| Tier | Hardware | Software / App | Runs in Repo | Real Data Validation | SIM Ground Truth Accuracy |
|---|---|---|---|---|---|
| **LiDAR** | iPhone 12 Pro, 13 Pro, 14 Pro, 15 Pro, 16 Pro (all LiDAR-equipped Pro iPhones) | Record3D (export: Shareable/External with depth, confidence, odometry, imu) | **YES** (full pipeline) | **Verified on 3 supplied captures:**<br>• `single_room`: 1 room, 4 walls, 11.8 m², ceiling observed<br>• `single_scan_floor_only`: 8 rooms, 59 m², ceiling honestly abstained, drift ablation evaluated<br>• `single_scan_with_ceiling`: multi-room stitched plan, 31.2s runtime | • **Ceiling height**: MAE 0.6 cm, max \|err\| 0.8 cm, **100% coverage**<br>• **Opening width**: MAE 0.5 cm, max \|err\| 0.8 cm, **100% coverage**<br>• **Wall length**: **93.8% coverage** post-fix (passes ≥85% gate)<br>• **Floor area**: **100% coverage** post-fix |
| **Video** | iPhone 15+ (any non-Pro or Pro iPhone) | Stock Camera App (walkthrough clip) or Record3D pose sidecar | **YES** (`video_mono` standalone and `video_pose`) | **Verified on walkthrough MP4s (`out/video_demo/`):**<br>• Optical flow tracking + essential matrix 5-point VO<br>• Manhattan frame alignment via horizontal vanishing lines<br>• Walking speed scale anchor ($v \approx 1.0\text{ m/s}$)<br>• Emits schema-valid `plan.json` + `plan.png` + `plan.svg` | Prior profile: 4.0 cm length abs, 4.0 cm ceiling abs, 6.0 cm opening abs |
| **Photo** | iPhone 15+ (any non-Pro or Pro iPhone) | Stock Camera App (2–8 stills per room, one folder per room) | **YES** (`photo` tier + multi-room stitcher) | **Verified on multi-room photo sets (`out/photo_demo/`):**<br>• LSD line segment detection + vanishing point horizon/pitch<br>• Room dimension extraction with chest-height camera prior ($1.45\text{ m}$)<br>• Whole-property assembly via beam-search door graph (`roomscan/stitch.py`)<br>• Emits schema-valid `plan.json` + `plan.png` + `plan.svg` | Prior profile: 6.0 cm length abs, 8.0 cm ceiling abs, 10.0 cm opening abs, 40 cm unobserved edge |

### Head-to-Head vs. Commercial Incumbent (Magicplan v12)
* **Dataset**: Evaluated on apartment benchmark rooms against Magicplan export (`examples/head_to_head/head_to_head.json`).
* **Result**: **84.6% beat-or-tie fraction** (11 wins out of 13 shared dimensions, with $\pm 5\text{ mm}$ tie threshold). Target was $\ge 70\%$.

### Repeatability Gate
* **Dataset**: Pairwise comparison of two independent captures of the same room (`examples/repeatability/repeatability.json`).
* **Result**: Ceiling height spread across captures $\le 2.7\text{ mm}$ (gate requirement $\le 10\text{ mm}$). Wall repeatability $\le 1\text{ cm}$ or $0.5\%$ on fully observed walls.


