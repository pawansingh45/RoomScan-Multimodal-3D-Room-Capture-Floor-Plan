"""Top-level orchestration: tier auto-detection, fallbacks, always-valid output."""
import os, json, time, logging, traceback
import numpy as np

from .config import Config
from .schema import validate_plan
from .render import render_plan
from .io.record3d import is_record3d

log = logging.getLogger("roomscan")
IMG_EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff")
VID_EXT = (".mp4", ".mov", ".m4v")


def detect_tier(path):
    if os.path.isfile(path):
        return "video" if path.lower().endswith(VID_EXT) else "photo"
    if is_record3d(path):
        return "lidar"
    names = os.listdir(path)
    if any(n.lower().endswith(VID_EXT) for n in names):
        return "video"
    if any(n.lower().endswith(IMG_EXT) for n in names) or any(
            os.path.isdir(os.path.join(path, n)) and any(m.lower().endswith(IMG_EXT) for m in os.listdir(os.path.join(path, n)))
            for n in names):
        return "photo"
    raise ValueError(f"cannot determine capture tier for {path}")


def empty_plan(tier, status, reason, source=""):
    iv = dict(value=None, lo=None, hi=None, unit="m2", level=0.9)
    return dict(schema="roomscan.plan/1.0", capture=dict(tier=tier, status=status, source=source, reason=reason),
                rooms=[], stitched_plan=dict(n_rooms=0, connections=[], footprint_area=iv, overlap_free=True),
                diagnostics=dict(error=reason))


def run(path, out_dir, tier="auto", cfg=None, drift_mode=None, render=True, damage=True):
    cfg = cfg or Config()
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    plan = None
    try:
        tier_req = detect_tier(path) if tier == "auto" else tier
        log.info("tier: %s", tier_req)
        if tier_req == "lidar":
            from .tiers.lidar import run_lidar
            plan, extra = run_lidar(path, cfg, drift_mode, keep_cloud=damage)
            if damage and extra is not None:
                try:
                    from .damage import run_damage
                    run_damage(plan, extra, cfg)
                except Exception as e:                       # damage is additive: never lose the geometry
                    log.warning("damage stage failed: %s", e)
                    plan["diagnostics"]["damage_error"] = repr(e)
        elif tier_req == "video":
            from .tiers.video import run_video
            plan = run_video(path, cfg, out_dir)
        elif tier_req == "photo":
            from .tiers.photo import run_photo
            plan = run_photo(path, cfg, out_dir)
        else:
            raise ValueError(f"unknown tier {tier_req}")
    except Exception as e:
        log.error("pipeline failed: %s\n%s", e, traceback.format_exc())
        plan = empty_plan(tier if tier != "auto" else "lidar", "failed", repr(e), str(path))
    errs = validate_plan(plan)
    if errs:
        log.error("output failed schema validation: %s", errs[:5])
        plan["diagnostics"]["schema_errors"] = errs[:20]
        plan["capture"]["status"] = "failed"
    plan["diagnostics"]["wall_clock_s"] = round(time.time() - t0, 2)
    with open(os.path.join(out_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if render and plan["rooms"]:
        try:
            render_plan(plan, os.path.join(out_dir, "plan.png"), os.path.join(out_dir, "plan.svg"), title=os.path.basename(os.path.normpath(path)))
        except Exception as e:
            log.warning("render failed: %s", e)
    return plan
