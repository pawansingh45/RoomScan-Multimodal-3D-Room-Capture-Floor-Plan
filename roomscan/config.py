"""All tunable parameters in one place. Override with a YAML file via --config."""
from dataclasses import dataclass, field, asdict, fields
import json, os


@dataclass
class Config:
    # ---- determinism / runtime ----
    seed: int = 0
    max_keyframes: int = 450          # cap on keyframes used for fusion
    kf_min_trans_m: float = 0.04      # keyframe selection thresholds
    kf_min_rot_deg: float = 4.0
    # ---- LiDAR depth filtering ----
    depth_min_m: float = 0.25
    depth_max_m: float = 4.5          # iPhone LiDAR is reliable to ~4-5 m
    min_confidence: int = 2           # Record3D confidence: 0 low / 1 med / 2 high
    edge_rel_thresh: float = 0.04     # drop flying pixels: |dz| > 4% of z between neighbours
    per_frame_voxel_m: float = 0.02
    global_voxel_m: float = 0.02
    # ---- drift handling ----
    drift_mode: str = "full"          # off | manhattan | full
    lc_min_dt_s: float = 12.0
    lc_max_dist_m: float = 2.0
    lc_icp_max_rms_m: float = 0.03
    lc_icp_min_overlap: float = 0.45
    # ---- geometry ----
    floor_band_m: float = 0.04
    wall_band_lo_m: float = 0.35      # wall evidence band above floor
    wall_band_hi_m: float = 1.85
    raster_res_m: float = 0.02
    min_room_area_m2: float = 1.0
    door_min_w: float = 0.55
    door_max_w: float = 1.60
    window_min_w: float = 0.35
    # ---- intervals ----
    ci_level: float = 0.90
    damage_detect: bool = False      # Optional image-based damage detector (opt in with --detect-damage)
    # ---- misc ----
    log_level: str = "INFO"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def load(path=None, **overrides):
        c = Config()
        if path:
            import yaml
            with open(path) as f:
                d = yaml.safe_load(f) or {}
            for k, v in d.items():
                if hasattr(c, k):
                    setattr(c, k, v)
        for k, v in overrides.items():
            if v is not None and hasattr(c, k):
                setattr(c, k, v)
        return c
