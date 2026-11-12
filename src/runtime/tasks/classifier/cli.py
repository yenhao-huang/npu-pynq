"""Run a compiled image classifier package on an image and record block times.

    python -m src.runtime.cli classifier --package package --image cat.jpg --labels labels.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

MODULE_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.runtime.share import profile
from src.runtime.tasks.classifier import ImageEngine, preprocess, top_predictions


def main(argv=None):
    parser = argparse.ArgumentParser(prog="src.runtime.cli classifier")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--backend", choices=("sim", "pynq"), default="pynq")
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--expected-class")
    parser.add_argument("--evidence", type=Path)
    profile.add_arguments(parser)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    times = {}
    t = time.perf_counter()
    overlay = None
    if args.backend == "pynq":
        if args.overlay is None:
            parser.error("--overlay is required for pynq")
        from pynq import Overlay
        overlay = Overlay(str(args.overlay))
    times["overlay"] = time.perf_counter() - t
    t = time.perf_counter()
    engine = ImageEngine(args.package, args.backend, overlay=overlay)
    times["load_package"] = time.perf_counter() - t
    t = time.perf_counter()
    from PIL import Image
    with Image.open(args.image) as image:
        tensor = preprocess(image, engine.contract)
    times["preprocess"] = time.perf_counter() - t
    profiler = profile.from_arguments(args, engine.package, task='classifier', package=str(args.package),
                                      backend=args.backend, image=str(args.image))
    t = time.perf_counter()
    logits = engine.predict(tensor, profiler)
    times["inference"] = time.perf_counter() - t
    profile.finish(profiler, args)
    t = time.perf_counter()
    labels = args.labels.read_text(encoding="utf-8").splitlines()
    predictions = top_predictions(logits, labels)
    times["prediction"] = time.perf_counter() - t
    times["total"] = time.perf_counter() - started
    if args.backend == "pynq" and engine.last_stats["jobs"] <= 0:
        raise RuntimeError("No physical NPU jobs executed")
    record = {"backend": args.backend, "package": str(args.package),
              "image": str(args.image), "block_seconds": times,
              "npu": engine.last_stats, "top5": [vars(p) for p in predictions],
              "expected_class": args.expected_class,
              "correct": predictions[0].name == args.expected_class if args.expected_class else None}
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if args.expected_class and not record['correct']:
        raise RuntimeError('Image prediction does not match the expected class')
    return 0

