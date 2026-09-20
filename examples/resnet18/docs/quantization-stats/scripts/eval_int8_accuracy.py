"""Measure FP32 versus INT8 accuracy on real ImageNet photographs.

This is the reproduction path behind the numbers in
``docs/exp/2026-09-20-int8-quantization-eval.md`` and the accuracy A/B table in
``examples/resnet18/README.md``. It uses the repository's own exporter to build
the quantized graph, and ``src/test/model/quantized_graph_reference.py`` to
execute it, which is pinned bit-exact to the scalar golden operators by
``test_vectorized_reference_matches_approved_scalar_operators``.

Absolute accuracy figures produced here are not comparable to published
ImageNet-1K validation numbers, because the corpus is a curated one-per-class
sample rather than the validation set. Same-image FP32-versus-INT8 deltas are
meaningful; absolute values are not.

Usage (relative to the repository root):

    python examples/resnet18/docs/quantization-stats/scripts/eval_int8_accuracy.py \\
        --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \\
        --corpus    examples/resnet18/model/imagenet-calibration \\
        --index     examples/resnet18/model/imagenet-calibration.index.tsv \\
        --limit     1000 \\
        --calibration synthetic \\
        --output    examples/resnet18/model/int8-eval/result.json

``--calibration synthetic`` reproduces the current on-disk calibration; ``real``
draws a disjoint slice of the same corpus, and the size is configurable via
``--calibration-size``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.export.torchvision_resnet18 import (
    build_quantized_resnet18,
    generate_calibration_inputs,
    load_checkpoint,
)
from src.test.model.quantized_graph_reference import execute_quantized_graph_reference


MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def _round_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def _preprocess(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for preprocessing") from error
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = 256 / min(width, height)
    image = image.resize(
        (round(width * scale), round(height * scale)), Image.BILINEAR
    )
    width, height = image.size
    left, top = (width - 224) // 2, (height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype=np.float32)


def _load_samples(corpus: Path, index_path: Path, limit: int):
    rows = [line.rstrip("\n").split("\t") for line in index_path.read_text().splitlines()]
    samples = [(int(r[0]), r[1], corpus / r[3]) for r in rows if (corpus / r[3]).exists()]
    return samples[: limit or len(samples)]


def _build_graph(checkpoint: Path, calibration: str, size: int, corpus: Path, index_path: Path):
    state = load_checkpoint(checkpoint)
    if calibration == "synthetic":
        inputs = generate_calibration_inputs()
        note = "2 synthetic images (linear gradient + checkerboard)"
    elif calibration == "real":
        rows = [line.rstrip("\n").split("\t") for line in index_path.read_text().splitlines()]
        real_paths = [corpus / r[3] for r in rows[500 : 500 + size]]
        missing = [p for p in real_paths if not p.exists()]
        if missing:
            raise RuntimeError(f"missing calibration images: {missing[:3]} ...")
        inputs = np.concatenate([_preprocess(p) for p in real_paths], axis=0)
        note = f"{size} real images from the corpus, disjoint from the first 500 evaluated"
    else:
        raise ValueError(f"unsupported calibration mode: {calibration}")
    graph, _quantized_input, _trace, scales = build_quantized_resnet18(state, inputs)
    return graph, float(scales["input"]), note


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--calibration", choices=("synthetic", "real"), default="synthetic",
    )
    parser.add_argument("--calibration-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fp32-only", action="store_true",
        help="Skip the INT8 path (useful for cheap FP32-baseline runs)",
    )
    arguments = parser.parse_args()

    try:
        import torch
        from torchvision.models import resnet18
    except ImportError as error:
        raise RuntimeError("torch and torchvision are required") from error

    samples = _load_samples(arguments.corpus, arguments.index, arguments.limit)
    if not samples:
        raise RuntimeError("no evaluation images found")
    print(f"INFO: evaluating {len(samples)} images", flush=True)

    model = resnet18()
    model.load_state_dict(
        torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()

    graph = constants = input_scale = calibration_note = None
    if not arguments.fp32_only:
        graph, input_scale, calibration_note = _build_graph(
            arguments.checkpoint,
            arguments.calibration,
            arguments.calibration_size,
            arguments.corpus,
            arguments.index,
        )
        constants = {
            item.name: np.asarray(
                item.values, dtype=np.dtype(item.dtype)
            ).reshape(item.shape)
            for item in graph.constants
        }
        print(f"INFO: calibration = {calibration_note}", flush=True)
        print(f"INFO: input_scale = {input_scale:.6f}", flush=True)

    fp32_top1 = fp32_top5 = int8_top1 = int8_top5 = agreement = 0
    start = time.time()
    for step, (label, _wnid, path) in enumerate(samples, start=1):
        nchw = _preprocess(path)
        with torch.no_grad():
            logits_fp32 = model(torch.from_numpy(nchw)).numpy().reshape(-1)
        top5_fp32 = np.argsort(-logits_fp32, kind="stable")[:5]
        fp32_top1 += int(top5_fp32[0] == label)
        fp32_top5 += int(label in top5_fp32)

        if graph is not None:
            quantized = np.clip(
                _round_away_from_zero(nchw / input_scale), -127, 127
            ).astype(np.int8)
            nhwc = np.ascontiguousarray(np.transpose(quantized, (0, 2, 3, 1)))
            logits_int8 = np.asarray(
                execute_quantized_graph_reference(graph, constants, {"input": nhwc})["logits"]
            ).reshape(-1).astype(np.int32)
            top5_int8 = np.argsort(-logits_int8, kind="stable")[:5]
            int8_top1 += int(top5_int8[0] == label)
            int8_top5 += int(label in top5_int8)
            agreement += int(top5_int8[0] == top5_fp32[0])

        if step % 100 == 0:
            rate = step / (time.time() - start)
            print(
                f"  {step}/{len(samples)}  {rate:.2f} img/s  "
                f"fp32_top1={fp32_top1/step:.4f}"
                + (f"  int8_top1={int8_top1/step:.4f}" if graph is not None else ""),
                flush=True,
            )

    n = len(samples)
    summary = {
        "images": n,
        "corpus": str(arguments.corpus),
        "checkpoint": str(arguments.checkpoint),
        "calibration": calibration_note or "not applicable (FP32 only)",
        "fp32": {"top1": fp32_top1 / n, "top5": fp32_top5 / n},
        "int8": (
            None if graph is None
            else {"top1": int8_top1 / n, "top5": int8_top5 / n}
        ),
        "delta": (
            None if graph is None
            else {"top1": (int8_top1 - fp32_top1) / n,
                  "top5": (int8_top5 - fp32_top5) / n}
        ),
        "top1_agreement_int8_vs_fp32": (
            None if graph is None else agreement / n
        ),
        "elapsed_seconds": time.time() - start,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=1))
    print("---", flush=True)
    print(json.dumps(summary, indent=1))
    print(f"PASS: wrote {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
