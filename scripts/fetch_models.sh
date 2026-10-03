#!/usr/bin/env bash
# Optional: only for the video tier. Pre-downloads the metric depth model (needs network + ~1.3 GB).
pip install torch transformers pillow
python - <<'PY'
from transformers import pipeline
pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
print("model cached")
PY
