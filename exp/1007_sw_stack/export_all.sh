#!/bin/bash
# Build the board packages for SmolLM2, Qwen3 and ResNet-18 into $1
# (default build/exp-payload). Runs on the hosted exp-board export job and
# on a developer machine alike; only the Cortex-A9 libraries are linked.
set -euo pipefail
out=${1:-build/exp-payload}
mkdir -p "$out"
export PYTHONPATH=.
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen3-0.6B"):
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt"])
PY
python - <<'PY'
import hashlib, json, urllib.request
from pathlib import Path
spec = json.loads(Path("examples/resnet18/model-source.json").read_text())
dest = Path("examples/resnet18/model") / spec["filename"]
if not dest.exists():
    dest.write_bytes(urllib.request.urlopen(spec["url"], timeout=300).read())
assert hashlib.sha256(dest.read_bytes()).hexdigest() == spec["sha256"], "ResNet-18 checkpoint hash mismatch"
PY
python -m src.inference.cli export smollm --out "$out/smollm" --targets pynq --awq
python -m src.inference.cli export qwen3 --out "$out/qwen3" --targets pynq --bits 4 --group 64 --awq
python -m src.inference.cli export resnet18 --out "$out/resnet18" --targets pynq
python - "$out" <<'PY'
import sys
from pathlib import Path
import runpy
mod = runpy.run_path("exp/1007_sw_stack/host/resnet_eval.py", run_name="lib")
for _, path in mod["gallery"](Path(sys.argv[1]) / "gallery"):
    print("gallery", path.name)
PY
for p in smollm qwen3 resnet18; do rm -rf "$out/$p/ir"; done
du -sh "$out"/*
