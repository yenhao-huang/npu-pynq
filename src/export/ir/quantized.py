"""Quantized tensors as the quantize stage hands them to to_mlir.

The operators in src/export/compilers/mlir/ops consume these types; they
never quantize. The numerics that produce them live in src/export/quantize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.export.ir.options import QuantRecipe
    from src.export.ir.source import SourceModel


EPS = 1e-8


@dataclass
class QuantizedWeight:
    q: np.ndarray        # int8 [K, N]: K inputs, N outputs
    scale: np.ndarray    # f32 [N] (bits=8) or [K // group, N] (bits=4)
    bits: int = 8
    group: int = 0

    @property
    def k(self) -> int:
        return self.q.shape[0]

    @property
    def n(self) -> int:
        return self.q.shape[1]

    def dequantize(self) -> np.ndarray:
        if self.bits == 32:
            return self.q
        if self.bits == 8:
            return self.q.astype(np.float32) * self.scale[None, :]
        g = self.group
        return (self.q.astype(np.float32).reshape(self.k // g, g, self.n) * self.scale[:, None, :]).reshape(self.k, self.n)


@dataclass
class OnnxQuantWeight:
    """An INT8 weight kept exactly as an ONNX source quantized it: q [K, N]
    with one scale per output column (fused projections keep each original
    projection's scale)."""
    q: np.ndarray        # int8 [K, N]
    scale: np.ndarray    # f32 [N]


@dataclass
class QuantizedModel:
    """What the quantize stage hands to to_mlir: every array in final form."""
    source: "SourceModel"
    recipe: "QuantRecipe"        # resolved: the group size is the one actually used
    config: Any                  # the architecture's parsed config, layers already clipped
    params: dict[str, Any]       # flat names ('embed', 'l3.qkv', 'norm', ...) -> QuantizedWeight,
                                 # OnnxQuantWeight or np.ndarray
    meta: dict = field(default_factory=dict)
