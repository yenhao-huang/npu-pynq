"""Symmetric weight quantization shared by the compiler and the quantization
tools: per-output-channel INT8, group-wise INT4, or an FP32 passthrough.

The numerics match src/test/model and the NPU contraction in
src/export/compilers/mlir/ops/linear.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


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


def quantize_weight(w_out_in: np.ndarray, bits: int = 8, group: int = 128) -> QuantizedWeight:
    """Symmetric quantization of a torch-layout [out, in] weight."""
    w = np.asarray(w_out_in, np.float32).T.copy()  # [K, N]
    if bits == 32:
        # Unquantized reference path: FP32 weights, contraction on the CPU.
        return QuantizedWeight(w, np.ones(w.shape[1], np.float32), 32, 0)
    if bits == 8:
        scale = np.maximum(np.abs(w).max(0), EPS) / 127.0
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        return QuantizedWeight(q, scale.astype(np.float32), 8, 0)
    if bits != 4:
        raise ValueError("bits must be 8 or 4")
    k, n = w.shape
    if k % group:
        raise ValueError(f"K={k} is not a multiple of group {group}")
    grouped = w.reshape(k // group, group, n)
    scale = np.maximum(np.abs(grouped).max(1), EPS) / 7.0
    q = np.clip(np.round(grouped / scale[:, None, :]), -8, 7).astype(np.int8).reshape(k, n)
    return QuantizedWeight(q, scale.astype(np.float32), 4, group)
