"""Gate definitions transcribed from the assignment brief (Part 2). Edit here, nowhere else."""

GATES = {
    "opening_width":   dict(tol_m=0.02, min_fraction=0.85, note="<=2 cm on >=85% of openings; a missed and a phantom opening each count as a miss"),
    "ceiling_height":  dict(tol_m=0.015, spread_m=0.010, note="<=1.5 cm per room; spread across repeat captures <=1 cm; report bias vs spread"),
    "repeatability":   dict(tol_m=0.01, tol_rel=0.005, note="two captures of the same room, same tier agree within 1 cm or 0.5% per wall"),
    "drift":           dict(note="state method + ablation of stitched footprint on/off; 'poses used as-is' is an automatic fail"),
    "photo_stitch":    dict(footprint_rel=0.08, note="one stitched plan, correct adjacency, no overlaps, footprint within +/-8% with calibrated intervals"),
    "wall_length":     {"lidar": dict(tol_m=0.02, tol_rel=0.01), "video_pose": dict(tol_rel=0.03), "video_mono": dict(tol_rel=0.03),
                        "photo": dict(tol_rel=0.08)},
}
# Baseline LiDAR wall length tolerance: max(2 cm, 1% of length).
