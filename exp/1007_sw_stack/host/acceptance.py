"""Host acceptance for #119: export SmolLM2, Qwen3 and ResNet-18 and run them on
the host with the simulated NPU (the same packages' CPU code targets the host).

    python exp/1007_sw_stack/host/acceptance.py --json exp/1007_sw_stack/host/host_acceptance.json
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.inference.engine import LLMEngine  # noqa: E402

PROMPT = "What is the capital of France?"


def export(model: str, out: Path, *flags: str) -> dict:
    t = time.monotonic()
    subprocess.run([sys.executable, "-m", "src.inference.cli", "export", model, "--out", str(out),
                    "--targets", "host,pynq", *flags], check=True, cwd=REPO, capture_output=True)
    manifest = json.loads((out / "manifest.json").read_text())
    return {"export_seconds": round(time.monotonic() - t, 2), "weights_bytes": manifest["weights"]["bytes"],
            "npu_tasks": manifest["npu"]["tasks"], "npu_segments": manifest["npu"]["segments"],
            "cpu_linalg_ops": sum(manifest["cpu_linalg_ops"].values()),
            "pynq_library": (out / "pynq/model.so").exists()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(REPO / "build/host-acceptance"))
    ap.add_argument("--json")
    args = ap.parse_args()
    work = Path(args.work)
    report = {"host": f"{platform.system()} {platform.machine()}", "backend": "sim (C ISA simulator)", "models": {}}
    try:
        for name, flags in (("smollm", ("--awq",)), ("qwen3", ())):
            pkg = work / name
            entry = export(name, pkg, *flags)
            engine = LLMEngine(pkg, "sim")
            ids, text, stats = engine.generate(PROMPT, 24, chat=True)
            entry.update({"options": engine.meta["options"], "prompt": PROMPT, "output": text, **stats.as_dict()})
            report["models"][name] = entry
            print(name, json.dumps(entry, ensure_ascii=False), flush=True)
            shutil.rmtree(pkg)
        pkg = work / "resnet18"
        entry = export("resnet18", pkg)
        out = subprocess.run([sys.executable, "exp/1007_sw_stack/host/resnet_eval.py", str(pkg)], cwd=REPO,
                             check=True, capture_output=True, text=True).stdout
        rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
        entry["images"] = rows
        entry["top1_correct"] = f"{sum(r['top1_correct'] for r in rows)}/{len(rows)}"
        entry["mean_seconds"] = float(np.mean([r["host_seconds"] for r in rows]))
        report["models"]["resnet18"] = entry
        print("resnet18", json.dumps(entry), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
