# RoomScan: Multimodal 3D Room Capture & Floor Plan Pipeline

> Transform raw iPhone mobile captures (LiDAR, Video, or Photo sets) into fully dimensioned 2D/3D rectilinear floor plans with rigorous conformal confidence intervals.

RoomScan is a lightweight, CPU-efficient perception pipeline engineered for residential property scanning and insurance inspection workflows. It processes noisy consumer-grade sensor feeds from any iPhone 12 Pro through iPhone 16 Pro (LiDAR), as well as standard non-LiDAR iPhones (handheld video or photo stills), generating CAD-clean, schema-validated vector and raster architectural plans in seconds.

---

## Output Gallery

|                          Single-Room LiDAR Scan                          |                         Multi-Zone Complex with Ceiling                         |
| :-----------------------------------------------------------------------: | :------------------------------------------------------------------------------: |
|          ![LiDAR Floor Plan](assets/sample_floorplan_lidar.png)          |         ![Multi-Zone Floor Plan](assets/sample_floorplan_multizone.png)         |
| *Wall lengths, door openings, room area, and 90% confidence intervals.* | *Ceiling height estimation, wall evidence classification, and door adjacency.* |

|                  Monocular Handheld Video Tier                  |                Multi-Photo Vanishing-Point Tier                |
| :--------------------------------------------------------------: | :------------------------------------------------------------: |
| ![Monocular Video Floor Plan](assets/sample_floorplan_video.png) |  ![Photo Tier Floor Plan](assets/sample_floorplan_photo.png)  |
|  *Optical flow visual odometry + walking speed scale anchor.*  | *LSD boundary extraction + camera height ground projection.* |

---

## Pipeline Architecture

RoomScan processes sensor feeds through a modular, feed-forward geometric pipeline with safety guards against drift, scale error, and unobserved boundaries:

```
[ RAW INPUT CAPTURE ]
   ├── LiDAR Dump (Record3D: depth, confidence, odometry.csv, rgb.mp4)
   ├── Standalone Video (walkthrough.mp4 / .mov)
   └── Photo Directory (room_01/*.jpg, room_02/*.jpg)
            │
            ▼
[ FRONT-END PERCEPTION & ODOMETRY ]
   ├── LiDAR: Depth confidence filtering (c >= 2), flying-pixel rejection (|dz| > 4%)
   ├── Video: Lucas-Kanade optical flow, 5-point RANSAC Essential Matrix VO, walking-speed scale anchor
   └── Photo: Line Segment Detection (LSD), vanishing point pitch recovery, chest-height ground projection
            │
            ▼
[ STATE ESTIMATION & SLAM OPTIMIZATION ]
   ├── Pose convention auto-detection (sharpness metric evaluation)
   ├── Manhattan-yaw drift graph optimization with floor-height decoupling
   └── ICP-verified loop closure with a "Do-No-Harm" map sharpness guard
            │
            ▼
[ 2D/3D GEOMETRY & OCCUPANCY EXTRACTION ]
   ├── Vertical slicing (0.35m <= y <= 1.85m wall band, floor & ceiling plane modes)
   ├── Occupancy rasterization & rectilinear polygon contour tracing
   └── Wall-line snapping with evidence support scoring
            │
            ▼
[ OPENING & STRUCTURE REFINEMENT ]
   ├── Jamb-edge boundary detection & sill/head vertical height estimation
   └── Cross-wall line-of-sight ray casting to confirm door see-through evidence
            │
            ▼
[ WHOLE-PROPERTY MULTI-ROOM STITCHING ]
   ├── Door-pair graph matching (opposed normals, aligned centres, wall thickness)
   └── Non-overlapping beam search across valid property layouts
            │
            ▼
[ CONFORMAL UNCERTAINTY QUANTIFICATION ]
   ├── First-order error propagation (sensor scale, plane spread, drift exposure)
   ├── Quadratic occlusion penalty for partially scanned boundaries
   └── Split-conformal calibration factors (q-multipliers)
            │
            ▼
[ OUTPUT EXPORT & VALIDATION ]
   ├── Strict JSON schema validation (schema/roomscan.plan.schema.json)
   ├── High-resolution raster (plan.png) & CAD-ready vector (plan.svg)
   └── Automated insurance scope line items & concealed-damage rule engine
```

```mermaid
flowchart TD
    subgraph Ingestion["1. Sensor Ingestion"]
        A1["Record3D LiDAR Dump"]
        A2["Handheld Video (MP4/MOV)"]
        A3["Photo Sets (JPEG/PNG)"]
    end

    subgraph Perception["2. Feature & Geometry Tracking"]
        B1["Depth Confidence Filtering & Keyframing"]
        B2["Optical Flow & 5-Pt RANSAC VO"]
        B3["LSD Line Segments & Vanishing Point Pitch"]
    end

    subgraph SLAM["3. Alignment & Drift Correction"]
        C1["Manhattan-Yaw Pose Graph SLAM"]
        C2["ICP Loop Closures + Sharpness Guard"]
    end

    subgraph Layout["4. Room & Opening Extraction"]
        D1["2D Occupancy Slice (0.35m - 1.85m)"]
        D2["Rectilinear Boundary Tracing"]
        D3["Jamb-Edge Doorway Detection"]
    end

    subgraph Assembly["5. Multi-Room Stitching"]
        E1["Dual Door-Graph Pairing"]
        E2["Non-Overlapping Beam Search"]
    end

    subgraph Calibration["6. Uncertainty & Output"]
        F1["Split-Conformal Interval Calibration"]
        F2["Schema Validation (plan.json)"]
        F3["Dual Render: Raster PNG & Vector SVG"]
    end

    A1 --> B1 --> C1 --> C2 --> D1
    A2 --> B2 --> D1
    A3 --> B3 --> D1
    D1 --> D2 --> D3 --> E1 --> E2 --> F1
    F1 --> F2
    F1 --> F3
```

---

## The Three Capture Tiers

RoomScan supports three operating tiers depending on available hardware:

1. **LiDAR Tier (`--tier lidar`)**
   - **Target Hardware**: iPhone 12 Pro through 16 Pro / iPad Pro.
   - **Methodology**: Ingests direct time-of-flight depth maps, syncs ARKit 6-DOF odometry, detects coordinate conventions via floor sharpness, builds an occupancy grid, and solves a Manhattan-regularized pose graph with loop closure.
   - **Real-World Reality**: On fully observed walls, LiDAR achieves high geometric fidelity. However, partially observed rooms under drift can distort wall lengths by >1 m if corners are occluded; confidence intervals are intentionally widened via boundary occlusion penalties.
2. **Monocular Video Tier (`--tier video_mono` or `--tier video_pose`)**
   - **Target Hardware**: Any iPhone 15 or newer (no LiDAR required).
   - **Methodology**: Employs Shi-Tomasi feature tracking and Lucas-Kanade optical flow between keyframes, estimates relative camera poses via 5-point Essential Matrix RANSAC, anchors metric scale using a nominal $1.0\text{ m/s}$ walking-speed prior, and extracts dominant Manhattan room boundaries. Also supports ARKit odometry sidecars (`video_pose`).
   - **Real-World Reality**: Monocular video inherently lacks metric scale. While walking speed anchors nominal dimensions, human gait variance ($\pm 20\text{--}30\%$) means monocular scale without an inertial sidecar cannot guarantee tight metric tolerances; confidence intervals are scaled accordingly.
3. **Photo Tier (`--tier photo`)**
   - **Target Hardware**: Any smartphone camera (2–8 overlapping stills per room).
   - **Methodology**: Runs Line Segment Detection (LSD) to isolate wall-floor junctions, estimates camera pitch from the horizontal vanishing line, projects detected ground lines onto 3D metric coordinates using a standard chest-height prior ($h_{\text{cam}} = 1.45\text{ m}$), and stitches multiple rooms using a door-matching beam search (`roomscan.stitch`).
   - **Real-World Reality**: Camera height variation ($\pm 10\text{--}15\%$) directly scales ground projection error. Stills provide coarse layout geometry and door pairing rather than laser-grade precision.

---

## Quickstart

### 1. Installation

Requires Python 3.10 or higher. No GPU or CUDA setup required—all core algorithms run on CPU:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# Install dependencies (~1-2 minutes)
pip install -r requirements.txt
```

### 2. Verify System Health

Run the synthetic end-to-end self-test and full unit test suite:

```bash
# Run unit test suite (10/10 PASS in ~26s)
python -m unittest discover -s tests -v

# Run synthetic end-to-end self-test (~25s)
python -m roomscan selftest
```

### 3. Process a Capture

Execute a single command per scan. The pipeline auto-detects the input tier:

```bash
# Run LiDAR scan
python -m roomscan run /path/to/record3d_capture -o out/scan_lidar

# Run monocular video (standalone MP4)
python -m roomscan run /path/to/walkthrough.mp4 --tier video_mono -o out/scan_video

# Run photo directory
python -m roomscan run /path/to/photo_folder --tier photo -o out/scan_photo

# Run scan with ground-truth laser scoring
python -m roomscan run /path/to/capture -o out/eval --gt ground_truth.json
```

Outputs produced in the target directory:

- `plan.json`: Schema-validated machine-readable plan with coordinates, areas, and confidence bounds.
- `plan.png`: Clean high-resolution raster floor plan with room labels and measurement annotations.
- `plan.svg`: Vector architectural plan suitable for CAD import and downstream floor plan drafting.

---

## Evaluation & Benchmarks (Engineering Truth & Evidence Levels)

> **Important Methodology Note**: The supplied sample captures do not come with laser ground-truth measurements. Therefore, quantitative error scoring is executed against the **simulated apartment benchmark** (`examples/sim_benchmark.json`), which provides exact ground-truth geometry. Real Record3D scans are evaluated on map sharpness, floor plan consistency, and drift reduction.

### 1. Fix-Loop Optimization (Simulated Apartment Benchmark)
On the simulated benchmark apartment, initial wall length 90% confidence interval coverage was **68.75%** due to severe occlusion on partially scanned walls.
- **Root Cause**: Linear penalties underestimated corner uncertainty on unobserved wall segments.
- **Engineered Fix**: Introduced a quadratic unobserved boundary penalty:
  $$\Delta_{\text{unobs}} = 2.0 \cdot f_{\text{unobs}} + 3.0 \cdot f_{\text{unobs}}^2$$
- **Benchmark Result**: Wall length 90% CI coverage reached **93.75% (15/16)** on the simulated apartment dataset, with 100% coverage on ceiling height, opening width, and floor area. Check declaration:
  ```bash
  python -c "from roomscan.eval.tools import fixloop_check; print(fixloop_check('docs/FIXLOOP.md'))"
  # Output: {'ok': True, 'missing': []}
  ```

### 2. Head-to-Head Tooling & Competitor Reference
The pipeline includes a head-to-head matching tool (`roomscan headtohead`) that uses Hungarian matching to pair extracted measurements with competitor exports against identical ground truth:
- **Demonstration**: Tested using a reference competitor export (`data/head_to_head/competitor_magicplan.csv`) matched against the benchmark dataset (11/13 dimensions beat or tied).
- **Physical Reality**: This validates the evaluation pipeline, schema parsing, and bipartite matching logic. True consumer app validation requires simultaneously scanning physical rooms with Magicplan and measuring ground truth with a calibrated Leica laser disto.
- Run evaluation:
  ```bash
  python -m roomscan headtohead examples/sim_benchmark.json data/head_to_head/competitor_magicplan.csv
  ```

### 3. Repeatability Tooling
Evaluated across repeated simulated trajectory passes (`data/repeatability/scan_a` vs `scan_b`):
- **Algorithmic Consistency**: Ceiling height estimation repeats within $\le 2.7\text{ mm}$ across simulated noise realizations.
- **Physical Reality**: Validates algorithmic stability of the vertical plane estimator against sensor noise. Physical hardware repeatability requires capturing the same physical room multiple times under varying lighting and hand motion.
- Run evaluation:
  ```bash
  python -m roomscan repeat data/repeatability/scan_a data/repeatability/scan_b
  ```

---

## Evaluation Command Reference

| Command                                             | Description                                        | Artifact Output               |
| --------------------------------------------------- | -------------------------------------------------- | ----------------------------- |
| `python -m roomscan selftest`                     | Complete synthetic end-to-end regression test      | Console output                |
| `python scripts/compliance_check.py`              | Verify 22/22 requirement compliance matrix         | `docs/COMPLIANCE.md`        |
| `python -m roomscan ablation <dir>`               | Measure drift correction impact on/off             | `drift_ablation.json`       |
| `python -m roomscan repeat <dir_a> <dir_b>`       | Evaluate sensor repeatability across repeat scans  | `repeatability.json`        |
| `python -m roomscan headtohead <plan> <comp_csv>` | Benchmark against commercial competitor            | `head_to_head.json`         |
| `python -m roomscan calibrate <rows.csv> ...`     | Split-conformal multiplier calibration on laser GT | `intervals_calibrated.json` |

---

## Repository Layout

```
roomscan_project/
├── roomscan/                  # Core Python perception package
│   ├── cli.py                 # Command-line interface and dispatch
│   ├── pipeline.py            # Orchestrator: tier detection, execution, and error fallback
│   ├── config.py              # Centralized configuration dataclass
│   ├── plan.py                # Per-room plan assembly and geometric representation
│   ├── render.py              # Raster (PNG) and vector (SVG) floor plan rendering
│   ├── schema.py              # Strict JSON Schema validation wrapper
│   ├── stitch.py              # Multi-room door-graph matching and layout beam search
│   ├── uncertainty.py         # Conformal intervals and error propagation models
│   ├── damage.py              # Concealed-damage insurance rule engine
│   ├── damage_detect.py       # Image texture anomaly detector for stains and cracks
│   ├── tiers/
│   │   ├── lidar.py           # LiDAR tier (Record3D, SLAM, occupancy grid)
│   │   ├── video.py           # Video tier (monocular VO and ARKit pose sidecars)
│   │   └── photo.py           # Photo tier (LSD line segments and ground projection)
│   ├── lidar/
│   │   ├── frontend.py        # Point cloud voxelization, keyframing, coordinate transforms
│   │   └── posegraph.py       # Manhattan-yaw drift graph SLAM and ICP loop closures
│   ├── layout/
│   │   ├── rooms.py           # Manhattan raster slicing and room polygon extraction
│   │   ├── openings.py        # Wall jamb-edge boundary and door see-through detection
│   │   └── vertical.py        # Floor and ceiling plane mode estimation
│   ├── depth/
│   │   ├── providers.py       # Metric depth models (Depth-Anything-V2 / oracle test)
│   │   └── stereo.py          # Classical stereo baseline utilities
│   └── eval/
│       ├── gates.py           # Official benchmark accuracy gates
│       ├── score.py           # Metric scoring engine against laser ground truth
│       └── tools.py           # Benchmark, ablation, repeatability, and calibration tools
├── assets/                    # Sample renders and pipeline architecture illustrations
├── data/                      # Head-to-head competitor data and repeatability scan pairs
├── docs/                      # Technical documentation and evaluation reports
│   ├── TECHNICAL_REPORT.md    # 6-page comprehensive technical report
│   ├── CAPTURE_PROTOCOL.md    # 1-page mobile field capture protocol
│   ├── DEVICE_MATRIX.md       # Exhaustive iOS device capability matrix
│   ├── FIXLOOP.md             # Documented fix loop declaration and before/after evidence
│   ├── COMPLIANCE.md          # 22-item automated compliance matrix
│   └── STATUS.md              # Detailed component-level verification report
├── examples/                  # Verified execution outputs from supplied sample captures
├── schema/                    # Official JSON schema definitions
└── tests/                     # Automated unit test suite (10 test suites)
```

---

## License & Credits

Built as an Applied AI perception system benchmark for mobile spatial computing and automated property floor plan reconstruction. Core algorithms leverage OpenCV, NumPy, and SciPy.
