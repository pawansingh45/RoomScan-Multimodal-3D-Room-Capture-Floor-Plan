# Decisions log (why, with the evidence that forced each)
D1 Pure NumPy/SciPy/OpenCV core: no GPU, installs in minutes; deep models are optional plug-ins (none shipped).
D2 Pose convention detected geometrically (floor-histogram sharpness), not assumed.
D3 4-DoF pose graph with Manhattan-yaw priors: the floor_only sample shows ~8 deg of heading drift; stock trust-region solver stalled -> own sparse LM.
D4 Door closure only along collinear walls: closing across both axes filled corridors with virtual wall (found on SIM).
D5 Do-no-harm guard on the pose graph: SIM with noise-free depth made it apply a confident wrong 6.4 deg correction.
D6 Openings from depth discontinuities (jambs): wall-evidence thresholds gave -5..-6 cm width bias on SIM; discontinuity snapping gave 0.4-0.9 cm.
D7 Ceiling abstain rule (horizontal plane covering >=15 % of footprint), because floor-only scans must not yield numbers.
D8 Intervals are explicit priors flagged calibrated=false; conformal tool provided; SIM showed they under-cover partially observed rooms (open issue).
D9 Overlap check erodes rooms by 4 cm: slivers thinner than a wall are reported separately, not as overlaps.
