"""Quantize a source that is already quantized: keep its numbers.

An INT8 ONNX export fixes every weight's scale; the preserve recipe only
regroups what the graph fuses and dequantizes what runs in float, and never
requantizes.
"""

from __future__ import annotations

import numpy as np

from src.export.ir.data import OnnxQuantWeight


def dequantize(q: np.ndarray, zero, scale) -> np.ndarray:
    return (q.astype(np.float32) - zero) * scale


def fuse_columns(tensors, keys: list[str]) -> OnnxQuantWeight:
    """Concatenate INT8 [K, N_i] weights along N, each keeping its own
    per-tensor scale as per-column scales."""
    matrices = [tensors[key + '.weight_quantized'] for key in keys]
    scales = [np.full(matrix.shape[1], tensors[key + '.weight_scale'], np.float32)
              for key, matrix in zip(keys, matrices)]
    return OnnxQuantWeight(np.concatenate(matrices, axis=1), np.concatenate(scales))
