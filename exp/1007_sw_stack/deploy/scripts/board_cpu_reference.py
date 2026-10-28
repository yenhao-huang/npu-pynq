"""Run the exact ARM package's built-in CPU ISA simulator without FPGA access.

The ARM library contains both runtime backends. A package view aliases its existing
ARM library as the CPU host library without changing weights, layouts or kernels.
This provides a same-binary reference and does not require Mac model/source transfer.
"""
import ctypes
import json
import os
from pathlib import Path
import sys

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY.parents[2]))
from acceptance import generate, package_info, save, sha256
from src.inference.engine import LLMEngine

source = DEPLOY / "packages/smollm"
view = DEPLOY / "packages/smollm-cpu-reference"
view.mkdir(parents=True, exist_ok=True)
for name in ("weights.bin", "programs.bin", "programs.json", "tokenizer.json", "pynq"):
    link = view / name
    if not link.exists():
        link.symlink_to(source / name, target_is_directory=(source / name).is_dir())
manifest = json.loads((source / "manifest.json").read_text())
manifest["libraries"]["host"] = manifest["libraries"]["pynq"]
save(view / "manifest.json", manifest)
for dep in ("libm.so.6", "libgcc_s.so.1"):
    ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
out = DEPLOY / "data/reference"
save(out / "package.json", package_info(view))
engine = LLMEngine(view, backend="sim", stream_weights=True)
runtime = engine.model.lib.npu_rt_stats().contents
assert runtime.backend == 0 and not runtime.mmio, "Reference must not use MMIO."
save(out / "provenance.json", {"method": "ARM CPU ISA simulator in the identical model library",
                              "backend": runtime.backend, "mmio": None, "uid": os.geteuid(),
                              "library_sha256": sha256(source / manifest["libraries"]["pynq"]),
                              "source_manifest_sha256": sha256(source / "manifest.json")})
result = generate(engine, out, display_token=lambda text: print(text, flush=True))
assert engine.model.lib.npu_rt_stats().contents.backend == 0
print(json.dumps(result, indent=2), flush=True)
