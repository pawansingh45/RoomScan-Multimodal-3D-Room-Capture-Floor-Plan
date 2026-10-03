"""Surfaces, damage regions, concealed-damage rules and scope line items.

Processes detected surface damage across walls and ceilings, evaluates rule-based
concealed damage risks (e.g. upper-wall leaks, cavity moisture, shared-wall damage),
and synthesizes repair scope line items.

Input detections format:
  [{"room":"R1","surface":"R1.w2","class":"water_stain","area_m2":0.35,"height_range_m":[1.9,2.4],"confidence":0.8}, ...]
"""
import os, json, logging
import numpy as np

log = logging.getLogger("roomscan")

DAMAGE_CLASSES = ["water_stain", "mold", "crack", "peeling_paint", "bubbling_paint", "missing_material", "efflorescence", "sagging"]

# class -> default scope action, unit cost-free quantity basis
SCOPE_ACTIONS = {
    "water_stain": ("seal_and_repaint", "m2"), "mold": ("remediate_mold_and_repaint", "m2"), "crack": ("repair_crack_and_repaint", "m"),
    "peeling_paint": ("scrape_prime_repaint", "m2"), "bubbling_paint": ("open_inspect_dry_repaint", "m2"),
    "missing_material": ("patch_and_finish", "m2"), "efflorescence": ("clean_treat_repaint", "m2"), "sagging": ("replace_board", "m2"),
}


def build_surfaces(room):
    sid = room["id"]
    surf = [dict(id=f"{sid}.floor", kind="floor", area_m2=room["floor_area"]["value"])]
    ch = room["ceiling_height"]
    H = ch.get("value") if ch.get("value") else None
    if H:
        surf.append(dict(id=f"{sid}.ceiling", kind="ceiling", area_m2=room["floor_area"]["value"]))
    for w in room["walls"]:
        if w["length"]["value"] < 0.3:
            continue
        open_area = sum(o["width"]["value"] * ((o["height"] or {}).get("value") or 2.0) for o in room["openings"] if o["wall_id"] == w["id"])
        a = w["length"]["value"] * H - open_area if H else None
        surf.append(dict(id=w["id"], kind="wall", area_m2=a, length_m=w["length"]["value"], observed=w["observed"]))
    return surf


def rule_engine(room, damages):
    """Named rules; every flag records the rule that fired."""
    flags, k = [], 0
    H = (room["ceiling_height"].get("value") or 2.4)
    by_surface = {}
    for d in damages:
        by_surface.setdefault(d["surface_id"], []).append(d)
    def add(rule, d, why, conf):
        nonlocal k
        k += 1
        flags.append(dict(id=f"{room['id']}.c{k}", rule=rule, surface_id=d["surface_id"], confidence=conf, because=why, source_damage=d["id"]))
    for d in damages:
        hr = d.get("height_range_m")
        if d["class"] in ("water_stain", "bubbling_paint", "sagging") and hr and hr[0] > 0.75 * H:
            add("R1_upper_wall_or_ceiling_stain_suggests_leak_above", d, "staining in the top quarter of the wall / ceiling: pipe, roof or upstairs leak", min(0.9, d["confidence"]))
        if d["class"] in ("water_stain", "mold", "efflorescence", "peeling_paint") and hr and hr[1] < 0.45:
            add("R2_baseboard_zone_moisture_suggests_rising_damp_or_wall_cavity_wet", d, "damage confined to the lowest 45 cm", min(0.85, d["confidence"]))
        if d["class"] == "bubbling_paint":
            add("R3_bubbling_paint_implies_moisture_behind_surface", d, "bubbling paint = trapped moisture behind the film", d["confidence"])
        if d["class"] == "mold" and d.get("area_m2", 0) > 0.1:
            add("R4_visible_mold_implies_hidden_growth_in_cavity", d, "visible mold > 0.1 m2: inspect behind the finish", min(0.8, d["confidence"]))
        if d["class"] == "sagging":
            add("R5_sagging_board_implies_saturated_substrate", d, "sagging board is saturated", d["confidence"])
    # shared-wall correlation: damage on two adjacent surfaces of the same room within the same class
    cls_surfaces = {}
    for d in damages:
        cls_surfaces.setdefault(d["class"], set()).add(d["surface_id"])
    for c, ss in cls_surfaces.items():
        if c in ("water_stain", "mold") and len(ss) >= 2:
            add("R6_same_class_damage_on_multiple_surfaces_suggests_common_source", dict(surface_id=sorted(ss)[0], id=",".join(sorted(ss))), f"{c} on {len(ss)} surfaces", 0.6)
    return flags


def scope_items(room, damages, flags, surfaces):
    items, n = [], 0
    for d in damages:
        act, unit = SCOPE_ACTIONS.get(d["class"], ("inspect", "ea"))
        q = d.get("area_m2", 0) if unit == "m2" else d.get("length_m", 1.0)
        n += 1
        items.append(dict(id=f"{room['id']}.s{n}", surface_id=d["surface_id"], action=act, quantity=round(float(q), 3), unit=unit, driver=d["id"],
                          margin=1.1))
    for f in flags:
        n += 1
        items.append(dict(id=f"{room['id']}.s{n}", surface_id=f["surface_id"], action="open_up_and_inspect_concealed", quantity=1, unit="ea", driver=f["rule"]))
    return items


def run_damage(plan, extra, cfg):
    root = plan["capture"].get("source", "")
    det_path = os.path.join(root, "damage_detections.json") if root else ""
    dets = []
    if det_path and os.path.exists(det_path):
        with open(det_path) as f:
            dets = json.load(f)
    if not dets and extra is not None and getattr(cfg, "damage_detect", True):
        try:
            from .damage_detect import detect_damage
            dets = detect_damage(plan, extra, cfg)
            plan["diagnostics"]["damage_detector"] = "classical_color_texture_votes (n=%d)" % len(dets)
            if root:
                json.dump(dets, open(os.path.join(os.path.dirname(os.path.normpath(root)) if False else ".", ".roomscan_last_detections.json"), "w")) if False else None
        except Exception as e:
            plan["diagnostics"]["damage_detector"] = "failed: %r" % (e,)
    if "damage_detector" not in plan["diagnostics"]:
        plan["diagnostics"]["damage_detector"] = "external_file" if dets else "none"
    for r in plan["rooms"]:
        r["surfaces"] = build_surfaces(r)
        ds = []
        for i, d in enumerate([d for d in dets if d.get("room") == r["id"]], 1):
            if d["class"] not in DAMAGE_CLASSES or d.get("surface") not in {s["id"] for s in r["surfaces"]}:
                continue
            ds.append(dict(id=f"{r['id']}.d{i}", surface_id=d["surface"], **{"class": d["class"]}, extent=dict(area_m2=d.get("area_m2"), length_m=d.get("length_m"),
                       height_range_m=d.get("height_range_m")), confidence=float(d.get("confidence", 0.5)), area_m2=d.get("area_m2", 0.0),
                       height_range_m=d.get("height_range_m")))
        r["damage"] = [{k: v for k, v in d.items() if k not in ("area_m2", "height_range_m")} for d in ds]
        flags = rule_engine(r, ds)
        r["concealed_damage_flags"] = flags
        r["scope_items"] = scope_items(r, ds, flags, r["surfaces"])
