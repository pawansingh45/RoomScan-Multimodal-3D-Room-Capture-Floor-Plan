# Implementation plan (what was built, in what order, and what is left)

## Architecture: one geometry core, tiers are front-ends
posed depth frames -> pose graph (drift) -> fused cloud -> Manhattan rasters -> wall-driven room segmentation ->
rectilinear polygons snapped to wall lines -> per-room ceiling/floor -> openings (jamb discontinuities) ->
adjacency/stitched plan -> intervals -> JSON (validated) + render. LiDAR tier supplies posed depth directly. Video/photo
tiers (TODO) must produce the same `posed depth frames` so the whole back end is shared.

## Built (all under roomscan/)
1. io/record3d.py: loader, per-frame intrinsics, timestamp sync, geometric pose-convention detection.
2. lidar/frontend.py: keyframes, flying-pixel filter, per-keyframe Manhattan angle + floor height, jamb-edge extraction.
3. lidar/posegraph.py: 4-DoF graph, sparse LM solver, ICP loop closures with overlap/RMS/degeneracy checks.
4. tiers/lidar.py: do-no-harm guard (accept correction only if wall sharpness does not degrade).
5. layout/*: rasters, door closure on collinear walls only, polygon tracing, wall-line snapping, ceilings, openings.
6. plan.py, uncertainty.py, schema.py, render.py, pipeline.py (always emits schema-valid output), cli.py.
7. eval/*: gates (from the brief), Hungarian scoring (missed/phantom openings count as misses), benchmark, ablation, repeatability,
   split-conformal calibration, head-to-head, fix-loop check.   sim/: synthetic apartment with exact GT.   damage.py: rules + scope.

## Remaining work, in priority order (you)
1. **Capture your benchmark NOW** (see CAPTURE_PROTOCOL.md): multi-room set, furnished room with staged damage, same rooms at 3 tiers, repeat captures, laser GT. Nothing real can be claimed without it.
2. **Fix wall-length interval under-coverage** (STATUS item 1) using `roomscan calibrate` on that GT; tighten room completion.
3. **Photo + video tiers** (tiers/video.py, tiers/photo.py): metric depth per frame (pose sidecar + stereo, or a metric-depth model), then feed the shared back end. Photo stitch: door-graph assembly with non-overlap constraint.
4. Image damage detector -> `damage_detections.json`.
5. Glass/mirror handling; Round-1 schema adapter; 6-page report; commit history that is genuinely yours.

## Fix loop (25 % of score): how to use the tooling
Run `roomscan benchmark`, pick the single worst gate from benchmark.json, write docs/FIXLOOP.md (worst gate + failing number, root-cause
hypothesis + evidence, fix, predicted number), COMMIT it, then ship the fix, commit, and re-run `roomscan benchmark` on the pre-fix and
post-fix commits (`git stash`/tags). The repo ships a *real example of this loop's raw material*: the jamb-edge refinement fixed a
-5..-6 cm opening-width bias found on the simulator (see DECISIONS.md D6). That is NOT your submission; you need your own.
