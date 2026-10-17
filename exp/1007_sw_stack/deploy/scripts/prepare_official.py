"""Assemble reproducible operator sessions and ARM oracle deployment inputs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from acceptance import save, sha256

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
BUILD = ROOT / "downloads/ort-arm-build"
ORT_COMMIT = "5c1b7ccbff7e5141c1da7a9d963d660e5741c319"


def operators():
    from official_runtime import ONNXGraph
    from official_ort_ops import export_sessions
    export_sessions(ONNXGraph(ROOT / "packages/onnx-graph"),
                    ROOT / "packages/onnx-ops")


def arm_inputs():
    summary = json.loads((ROOT / "runs/official-reference/summary.json").read_text())
    if not summary["pass"] or summary["steps"] != 24:
        raise ValueError("Complete the passing 24-step host reference first")
    source = ROOT / "packages/onnx/model_int8.onnx"
    if sha256(source) != summary["source_sha256"]:
        raise ValueError("Host reference and model hashes differ")
    BUILD.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts/official_ort_full_reference.c", BUILD / "onnx_ort_full_reference.c")
    shutil.copy2(ROOT / "scripts/run_arm_ort_oracle.sh", BUILD / "run_arm_ort_oracle.sh")
    shutil.copy2(REPO / "src/runtime/c/onnx_ort_bridge.c", BUILD / "onnx_ort_bridge.c")
    shutil.copy2(REPO / "src/runtime/build_onnxruntime_arm.sh", BUILD / "build_ort_arm.sh")
    shutil.copy2(source, BUILD / source.name)
    np.asarray(summary["prompt_ids"], dtype="<i8").tofile(BUILD / "prompt_ids.bin")


def arm_reference():
    host = ROOT / "runs/official-reference"
    summary = json.loads((host / "summary.json").read_text())
    prompt = np.fromfile(BUILD / "prompt_ids.bin", dtype="<i8")
    ids = np.fromfile(BUILD / "oracle-arm.ids.bin", dtype="<i8")
    logits = np.fromfile(BUILD / "oracle-arm.logits.bin", dtype="<f4")
    if prompt.tolist() != summary["prompt_ids"] or len(ids) != 24:
        raise ValueError("ARM oracle prompt or step count differs")
    if logits.size != 24 * 49152 or not np.isfinite(logits).all():
        raise ValueError("ARM oracle logits are incomplete or nonfinite")
    logits = logits.reshape(24, 49152)
    if not np.array_equal(ids, logits.argmax(axis=1)):
        raise ValueError("ARM oracle IDs do not match its logits")
    if sha256(BUILD / "model_int8.onnx") != summary["source_sha256"]:
        raise ValueError("ARM oracle model differs from the host reference")
    target = ROOT / "data/official-arm-oracle"
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "reference.npz", prompt_ids=prompt, ids=ids, logits=logits)
    save(target / "summary.json", {
        "source_sha256": summary["source_sha256"], "runtime": "1.20.1",
        "architecture": "Cortex-A9 ARMv7 QEMU", "unchanged_onnx": True,
        "steps": 24, "prompt_tokens": len(prompt), "prompt_ids": prompt.tolist(),
        "ids": ids.tolist(), "prompt_ids_sha256": sha256(BUILD / "prompt_ids.bin"),
        "reference_sha256": sha256(target / "reference.npz"),
        "runtime_sha256": sha256(BUILD / "libonnxruntime.so.1.20.1"),
        "oracle_source_sha256": sha256(BUILD / "onnx_ort_full_reference.c"),
        "ort_source_commit": ORT_COMMIT,
    })
    host_target = ROOT / "data/official-oracle"
    host_target.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "reference.npz"):
        shutil.copy2(host / name, host_target / name)
    runtime = ROOT / "packages/onnx-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ("libonnxruntime.so.1.20.1", "onnxruntime_c_api.h"):
        shutil.copy2(BUILD / name, runtime / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("operators", "arm-inputs", "arm-reference"))
    args = parser.parse_args()
    {"operators": operators, "arm-inputs": arm_inputs,
     "arm-reference": arm_reference}[args.phase]()
