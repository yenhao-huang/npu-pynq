#!/bin/bash
# Build the SmolLM2-135M-Instruct board package (W8A8 + AWQ, Cortex-A9 target
# only) into $1/smollm (default build/exp-payload), for the #136 exp-board
# run. Same steps as exp/1007_sw_stack/export_all.sh, SmolLM2 only.
set -euo pipefail
out=${1:-build/exp-payload}
mkdir -p "$out"
export PYTHONPATH=.
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("HuggingFaceTB/SmolLM2-135M-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt"])
PY
python -m src.inference.cli export smollm --out "$out/smollm" --targets pynq --awq
rm -rf "$out/smollm/ir"
du -sh "$out"/*
