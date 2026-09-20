"""Dump the per-layer weight distribution that INT8 quantization actually sees.

The tensor that gets quantized is not the checkpoint weight: the exporter folds
BatchNorm into the convolution first (``_folded_convolutions`` and
``fold_batch_norm``), then quantizes per output channel with
``scale[o] = max|W[o]| / 127``.

This script writes:

- ``<out_prefix>.json`` — per-layer summary, including raw checkpoint and
  folded max, per-output-channel max min/median/max, and the relative
  reconstruction error under per-channel and per-tensor quantization.
- ``<out_prefix>.channels.csv`` — one row per output channel:
  ``layer, channel, max_abs, scale``.

Feed the JSON to ``build_weight_viz.py`` to render an HTML distribution page.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.export.torchvision_resnet18 import (
    _conv_specs,
    _folded_convolutions,
    _round_away_from_zero,
    load_checkpoint,
    quantize_conv_weight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    arguments = parser.parse_args()

    state = load_checkpoint(arguments.checkpoint)
    folded = _folded_convolutions(state)
    specs = {spec.command_id: spec for spec in _conv_specs()}

    layers = []
    channel_rows = []
    for name in folded:
        weight, _bias = folded[name]
        raw = state[f"{specs[name].source_prefix}.weight"]

        per_channel_max = np.max(np.abs(weight), axis=(1, 2, 3))
        qweight, used_scales = quantize_conv_weight(weight)
        dequantized = (
            np.transpose(qweight, (3, 2, 0, 1)).astype(np.float64)
            * used_scales[:, None, None, None]
        )
        norm_weight = max(np.linalg.norm(weight), 1e-12)
        err_channel = float(np.linalg.norm(weight - dequantized) / norm_weight)

        tensor_scale = np.max(np.abs(weight)) / 127.0
        dq_tensor = (
            _round_away_from_zero(np.clip(weight / tensor_scale, -127, 127))
            * tensor_scale
        )
        err_tensor = float(np.linalg.norm(weight - dq_tensor) / norm_weight)

        layers.append({
            "layer": name,
            "shape_OIHW": list(map(int, weight.shape)),
            "parameters": int(weight.size),
            "raw_checkpoint": {
                "max_abs": float(np.abs(raw).max()),
                "std": float(raw.std()),
            },
            "folded": {
                "max_abs": float(np.abs(weight).max()),
                "std": float(weight.std()),
                "max_over_std": float(
                    np.abs(weight).max() / max(weight.std(), 1e-12)
                ),
            },
            "per_output_channel_max_abs": {
                "min": float(per_channel_max.min()),
                "median": float(np.median(per_channel_max)),
                "max": float(per_channel_max.max()),
                "max_over_min": float(
                    per_channel_max.max() / max(per_channel_max.min(), 1e-12)
                ),
            },
            "relative_error": {
                "per_channel": err_channel,
                "per_tensor_if_used_instead": err_tensor,
            },
        })
        for index, value in enumerate(per_channel_max):
            channel_rows.append({
                "layer": name,
                "channel": index,
                "max_abs": float(value),
                "scale": float(used_scales[index]),
            })

    out_json = arguments.out_prefix.with_suffix(".json")
    out_csv = Path(str(arguments.out_prefix) + ".channels.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"layers": layers}, indent=1))
    with out_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["layer", "channel", "max_abs", "scale"]
        )
        writer.writeheader()
        writer.writerows(channel_rows)

    header = (
        f"{'layer':<22}{'folded max':>11}{'max/std':>9}"
        f"{'ch max/min':>12}{'err ch':>9}{'err tensor':>11}"
    )
    print(header)
    for row in layers:
        print(
            f"{row['layer']:<22}{row['folded']['max_abs']:>11.4f}"
            f"{row['folded']['max_over_std']:>9.1f}"
            f"{row['per_output_channel_max_abs']['max_over_min']:>12.1f}"
            f"{row['relative_error']['per_channel']:>9.4f}"
            f"{row['relative_error']['per_tensor_if_used_instead']:>11.4f}"
        )
    print(f"PASS: wrote {out_json} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
