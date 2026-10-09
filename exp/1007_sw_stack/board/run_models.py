"""Run the exported SmolLM2, Qwen3 and ResNet-18 packages on the PYNQ-Z1 NPU.

    python run_models.py --overlay overlay/artifacts/npu_matrix.bit --packages export --out results/models.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def run_llm(name, package, overlay, prompt, max_new, report):
    from src.inference.engine import LLMEngine

    t = time.monotonic()
    engine = LLMEngine(package, "pynq", overlay=overlay)
    load = time.monotonic() - t
    ids, text, stats = engine.generate(prompt, max_new, chat=True)
    report[name] = {"prompt": prompt, "output": text, "output_ids": ids, "load_seconds": load,
                    "stream_weights": engine.model.stream_weights, **stats.as_dict()}
    print(name, json.dumps(report[name], ensure_ascii=False), flush=True)
    return bool(ids)


def run_resnet(package, overlay, gallery, report):
    from PIL import Image
    from src.models.resnet18 import model as R
    from src.runtime.compiled import CompiledModel

    labels = json.loads((Path(package) / "labels.json").read_text()) if (Path(package) / "labels.json").exists() else None
    model = CompiledModel(package, "pynq", overlay=overlay)
    rows = []
    for path in sorted(Path(gallery).glob("*.jpg")):
        x = R.preprocess(Image.open(path))
        t = time.monotonic()
        logits = R.classify(model, x)
        top5 = [int(i) for i in np.argsort(-logits)[:5]]
        rows.append({"image": path.name, "top5": top5, "top5_labels": [labels[i] for i in top5] if labels else None,
                     "seconds": time.monotonic() - t})
        print("resnet18", json.dumps(rows[-1]), flush=True)
    report["resnet18"] = {"images": rows, "npu": model.stats()}
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--packages", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from pynq import Overlay

    overlay = Overlay(args.overlay)
    root = Path(args.packages)
    report, ok = {}, True
    jobs = [("resnet18", lambda: run_resnet(root / "resnet18", overlay, root / "gallery", report)),
            ("smollm", lambda: run_llm("smollm", root / "smollm", overlay, "What is the capital of France?", 24, report)),
            ("qwen3", lambda: run_llm("qwen3", root / "qwen3", overlay, "What is the capital of France?", 12, report))]
    for name, job in jobs:
        try:
            ok &= bool(job())
        except Exception as error:  # keep going: one model's failure must not hide the others
            ok = False
            report[name] = {"error": repr(error), "traceback": traceback.format_exc()}
            print(name, "FAILED", repr(error), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
