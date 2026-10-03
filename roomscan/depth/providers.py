"""Metric depth providers for monocular depth estimation.

Supports:
  - hf: Depth-Anything-V2-Metric-Indoor via HuggingFace transformers pipeline.
  - oracle: Ground truth degraded with calibrated noise and scale drift for testing.
"""
import os, logging
import numpy as np
import cv2

log = logging.getLogger("roomscan")


class HFMetricDepth:
    name = "hf_depth_anything_v2_metric_indoor"

    def __init__(self, model=None, device=None):
        import torch
        from transformers import pipeline
        self.model_id = model or os.environ.get("ROOMSCAN_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
        dev = device or ("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
        self.pipe = pipeline("depth-estimation", model=self.model_id, device=dev)

    def predict(self, bgr, meta=None):
        from PIL import Image
        out = self.pipe(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        d = np.asarray(out["predicted_depth"] if "predicted_depth" in out else out["depth"], np.float32)
        if d.shape[:2] != bgr.shape[:2]:
            d = cv2.resize(d, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return d      # metres


class OracleDegraded:
    name = "oracle_degraded (DEV ONLY)"

    def __init__(self, ref_capture, scale_sigma=0.04, noise=0.015, seed=0):
        self.cap, self.ss, self.noise, self.rng = ref_capture, scale_sigma, noise, np.random.default_rng(seed)

    def predict(self, bgr, meta):
        d = self.cap.depth_m(meta["cap_idx"])
        if d is None:
            return None
        d = cv2.GaussianBlur(d, (5, 5), 0) * (1 + self.rng.normal(0, self.ss))
        return (d + self.rng.normal(0, self.noise, d.shape).astype(np.float32) * (d > 0)).astype(np.float32)


def auto_provider():
    try:
        return HFMetricDepth()
    except Exception as e:
        log.warning("No learned depth provider available (%s)", type(e).__name__)
        return None
