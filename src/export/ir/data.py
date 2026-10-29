"""Data handed between the front-end stages.

    SourceModel     load -> quantize: a source read and verified, numbers unchanged
    QuantizedModel  quantize -> to_mlir: every parameter in final form
    QuantizedWeight / OnnxQuantWeight: the quantized matrices inside it

The operators in src/export/compilers/mlir/ops consume these types; they
never quantize. The numerics that produce them live in src/export/quantize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.export.ir.options import QuantRecipe


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
class SourceModel:
    arch: str                    # model_arch registry name, e.g. 'smollm' or 'smollm_onnx'
    format: str                  # 'safetensors' | 'onnx' | 'torch'
    source: str                  # as the caller named it; recorded as meta['source']
    root: Path                   # directory holding the tokenizer and config
    config: dict                 # raw config.json ({} when the source has none)
    tensors: Any                 # name -> np.ndarray: SafeTensors, OnnxWeights or a dict
    layers: int | None = None    # debug: keep only the first N decoder layers
    provenance: dict = field(default_factory=dict)   # digests, pins, imported ONNX graph


@dataclass
class QuantizedModel:
    """What the quantize stage hands to to_mlir: every array in final form."""
    source: "SourceModel"
    recipe: "QuantRecipe"        # resolved: the group size is the one actually used
    config: Any                  # the architecture's parsed config, layers already clipped
    params: dict[str, Any]       # flat names ('embed', 'l3.qkv', 'norm', ...) -> QuantizedWeight,
                                 # OnnxQuantWeight or np.ndarray
    meta: dict = field(default_factory=dict)
