"""Quantized linear layers: activation quantization on the CPU, the INT8
contraction on the NPU, dequantization fused back on the CPU.

W8A8: per-output-channel INT8 weights, per-row (per-token) INT8 activations.
W4A8: group-wise INT4 weights (values in [-8, 7], one scale per K group of
``group`` rows and output channel). Each group's INT32 partial product comes
back separately (npu.matmul planes), so per-group scales stay exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.compiler.mlir_builder import Region, Value, identity_map, map_of
from src.compiler.ops import core
from src.compiler.ops.context import Ctx

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


def quantize_rows(ctx: Ctx, x: Value) -> tuple[Value, Value]:
    """Per-row symmetric INT8: returns (q [.., K] i8, scale [..] f32)."""
    b = ctx.b
    amax = core.reduce_last(b, x, "absmax")
    scale = b.elementwise([amax], "f32", lambda r: r.mulf(r.maxf(r.args[0], r.const(EPS, "f32")),
                                                          r.const(1.0 / 127.0, "f32")))
    rank = x.rank
    dims = [f"d{i}" for i in range(rank)]

    def body(r: Region) -> Value:
        inv = r.divf(r.const(1.0, "f32"), r.args[1])
        v = r.roundeven(r.mulf(r.args[0], inv))
        v = r.minf(r.maxf(v, r.const(-127.0, "f32")), r.const(127.0, "f32"))
        return r.fptosi(v, "i8")

    q = b.elementwise([x, scale], "i8", body, maps=[identity_map(rank), map_of(rank, dims[:-1])])
    return q, scale


def linear(ctx: Ctx, x: Value, name: str, w: QuantizedWeight, bias: np.ndarray | None = None) -> Value:
    """y[M, N] = x[M, K] @ W + bias, with the contraction on the NPU."""
    b = ctx.b
    m, k = x.shape
    if k != w.k:
        raise ValueError(f"{name}: x has K={k}, weight K={w.k}")
    n = w.n
    if w.bits == 32:
        wf = ctx.weight(f"{name}.w", w.q)
        y = b.matmul(x, wf, b.fill(0.0, [m, n], "f32"))
        if bias is None:
            return y
        bv = ctx.weight(f"{name}.b", np.asarray(bias, np.float32))
        return b.elementwise([y, bv], "f32", lambda r: r.addf(r.args[0], r.args[1]),
                             maps=[identity_map(2), map_of(2, ["d1"])])
    xq, xs = quantize_rows(ctx, x)
    wq = ctx.weight(f"{name}.q", w.q)
    ws = ctx.weight(f"{name}.s", w.scale)
    bv = ctx.weight(f"{name}.b", np.asarray(bias, np.float32)) if bias is not None else None
    if w.bits == 8:
        acc = b.matmul(xq, wq, b.fill(0, [m, n], "i32"))
        inputs = [acc, xs, ws] + ([bv] if bv is not None else [])
        maps = [identity_map(2), map_of(2, ["d0"]), map_of(2, ["d1"])] + ([map_of(2, ["d1"])] if bv is not None else [])

        def body(r: Region) -> Value:
            y = r.mulf(r.mulf(r.sitofp(r.args[0]), r.args[1]), r.args[2])
            return r.addf(y, r.args[3]) if bv is not None else y

        return b.elementwise(inputs, "f32", body, shape=[m, n], maps=maps)

    g = w.group
    groups = k // g
    acc = b.fill(0, [groups, m, n], "i32")
    # C[c, m, n] = sum_j X[m, c, j] * W[c, j, n] on (c, m, n, j).
    x3 = b.reshape(xq, [m, groups, g])
    w3 = ctx.weight(f"{name}.q", shape=[groups, g, n], dtype="i8")
    maps = [map_of(4, ["d1", "d0", "d3"]), map_of(4, ["d0", "d3", "d2"]), map_of(4, ["d0", "d1", "d2"])]

    def mac(r: Region) -> list[Value]:
        p = r.muli(r.extsi(r.args[0]), r.extsi(r.args[1]))
        return [r.addi(p, r.args[2])]

    partials = b.generic([x3, w3], [acc], maps, ["parallel", "parallel", "parallel", "reduction"], mac,
                         attrs=f"npu.group_size = {g} : i64")[0]
    # y[m, n] = xs[m] * sum_c partial[c, m, n] * ws[c, n]
    out = b.fill(0.0, [m, n], "f32")

    def group_sum(r: Region) -> list[Value]:
        return [r.addf(r.mulf(r.sitofp(r.args[0]), r.args[1]), r.args[2])]

    summed = b.generic([partials, ws], [out], [map_of(3, ["d2", "d0", "d1"]), map_of(3, ["d2", "d1"]),
                                                map_of(3, ["d0", "d1"])],
                       ["parallel", "parallel", "reduction"], group_sum)[0]
    inputs = [summed, xs] + ([bv] if bv is not None else [])
    maps = [identity_map(2), map_of(2, ["d0"])] + ([map_of(2, ["d1"])] if bv is not None else [])

    def finish(r: Region) -> Value:
        y = r.mulf(r.args[0], r.args[1])
        return r.addf(y, r.args[2]) if bv is not None else y

    return b.elementwise(inputs, "f32", finish, shape=[m, n], maps=maps)


def ref_quantize_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amax = np.abs(x).max(-1)
    scale = (np.maximum(amax, EPS) * np.float32(1.0 / 127.0)).astype(np.float32)
    q = np.clip(np.round(x * (1.0 / scale)[..., None]), -127, 127).astype(np.int8)
    return q, scale


def ref_linear(x: np.ndarray, w: QuantizedWeight, bias: np.ndarray | None = None) -> np.ndarray:
    if w.bits == 32:
        y = x.astype(np.float32) @ w.q
        return (y + bias[None, :] if bias is not None else y).astype(np.float32)
    xq, xs = ref_quantize_rows(x.astype(np.float32))
    if w.bits == 8:
        acc = xq.astype(np.int32) @ w.q.astype(np.int32)
        y = acc.astype(np.float32) * xs[:, None] * w.scale[None, :]
    else:
        g = w.group
        parts = np.einsum("mcj,cjn->cmn", xq.reshape(xq.shape[0], -1, g).astype(np.int64),
                          w.q.reshape(-1, g, w.n).astype(np.int64)).astype(np.float32)
        y = (parts * w.scale[:, None, :]).sum(0) * xs[:, None]
    if bias is not None:
        y = y + bias[None, :]
    return y.astype(np.float32)
