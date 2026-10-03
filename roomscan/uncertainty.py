"""Confidence intervals on every measurement.

sigma is built from named error sources (sensor scale, plane-fit spread, un-observed-wall inference, drift
exposure), then multiplied by a per-tier/per-measure *conformal* factor q learned from ground truth
(split conformal on |error|/sigma; see eval/calibrate.py). Until calibrated on real GT the output says so.
"""
import json, os
import numpy as np
from scipy.stats import norm

_DEF = os.path.join(os.path.dirname(__file__), "data", "intervals_default.json")


class IntervalModel:
    def __init__(self, path=None, level=0.90):
        with open(path or _DEF) as f:
            self.p = json.load(f)
        self.level = level
        self.z = float(norm.ppf(0.5 + level / 2))

    def tier_key(self, tier):
        return tier if tier in self.p["tiers"] else "lidar"

    def _iv(self, value, sigma, tier, measure, unit="m", basis=""):
        q = self.p["q"][self.tier_key(tier)].get(measure, 1.0)
        half = self.z * sigma * q
        return dict(value=float(value), lo=float(value - half), hi=float(value + half), unit=unit,
                    level=self.level, sigma=float(sigma), basis=basis, calibrated=bool(self.p["calibrated"]))

    def wall_length(self, L, tier, observed_ends, spread=0.0, travelled_m=0.0, extra=0.0):
        t = self.p["tiers"][self.tier_key(tier)]
        s2 = t["len_abs"] ** 2 + (t["scale_rel"] * L) ** 2 + (t["drift_per_m"] * min(L, 6.0)) ** 2 + 2 * spread ** 2
        s2 += (observed_ends < 2) * t["unobs_edge"] ** 2 * (1 if observed_ends == 1 else 2) + extra ** 2
        return self._iv(L, np.sqrt(s2), tier, "length", basis="plane-fit + scale + drift" + ("" if observed_ends == 2 else " + unobserved-wall inference") + (" + partial-room penalty" if extra > 0 else ""))

    def ceiling(self, H, tier, plane_sd=0.0):
        t = self.p["tiers"][self.tier_key(tier)]
        s = np.sqrt(t["ceil_abs"] ** 2 + (t["scale_rel"] * H) ** 2 + plane_sd ** 2)
        return self._iv(H, s, tier, "ceiling", basis="floor/ceiling plane modes + scale")

    def opening_width(self, w, tier, confirmed=True):
        t = self.p["tiers"][self.tier_key(tier)]
        s = np.sqrt(t["open_abs"] ** 2 + (t["scale_rel"] * w) ** 2) * (1.0 if confirmed else 2.0)
        return self._iv(w, s, tier, "opening", basis="jamb-edge localisation" + ("" if confirmed else " (weak see-through evidence)"))

    def area(self, A, edges, tier):
        """edges: list of (length, normal-offset sigma). First-order propagation: dA = sum L_i * dn_i."""
        s = float(np.sqrt(sum((L * sn) ** 2 for L, sn in edges)))
        t = self.p["tiers"][self.tier_key(tier)]
        s = float(np.sqrt(s ** 2 + (2 * t["scale_rel"] * A) ** 2))
        return self._iv(A, s, tier, "area", unit="m2", basis="edge-offset propagation + scale")

    def edge_offset_sigma(self, tier, observed):
        t = self.p["tiers"][self.tier_key(tier)]
        return t["len_abs"] if observed else t["unobs_edge"]
