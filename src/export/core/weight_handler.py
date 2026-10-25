"""Turn one decoder layer's float weights into what the operators consume.

A model frontend picks its layer's projection matrices (torch [out, in]
layout, already fused where the model fuses them) and its norm vectors; this
applies the AWQ hook when given, then quantizes every matrix the same way.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.export.quant.int8 import QuantizedWeight, quantize_weight

AwqHook = Callable[[int, dict, dict], tuple[dict, dict]]


def quantize_layer(layer: int, mats: dict[str, np.ndarray], norms: dict[str, np.ndarray],
                   bits: int, group: int, awq: AwqHook | None = None) -> dict[str, QuantizedWeight | np.ndarray]:
    """Return ``{name: QuantizedWeight}`` for ``mats`` and ``{name: f32}`` for ``norms``."""
    if awq is not None:
        mats, norms = awq(layer, mats, norms)
    out: dict[str, QuantizedWeight | np.ndarray] = {k: quantize_weight(v, bits, group) for k, v in mats.items()}
    out.update({k: v.astype(np.float32) for k, v in norms.items()})
    return out
