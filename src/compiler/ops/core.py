"""Elementwise, broadcast and reduction building blocks."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.compiler.mlir_builder import FuncBuilder, Region, Value, identity_map, map_of


def _dims(rank: int) -> list[str]:
    return [f"d{i}" for i in range(rank)]


def binary(b: FuncBuilder, x: Value, y: Value, fn: str) -> Value:
    """Elementwise op; ``y`` may be a trailing-dims broadcast of ``x``."""
    rank = x.rank
    dims = _dims(rank)
    y_map = map_of(rank, dims[rank - y.rank:])
    ops = {"add": Region.addf, "sub": Region.subf, "mul": Region.mulf, "div": Region.divf}
    return b.elementwise([x, y], x.dtype, lambda r: ops[fn](r, r.args[0], r.args[1]),
                         maps=[identity_map(rank), y_map])


def add(b, x, y): return binary(b, x, y, "add")
def mul(b, x, y): return binary(b, x, y, "mul")
def sub(b, x, y): return binary(b, x, y, "sub")


def scale_rows(b: FuncBuilder, x: Value, s: Value) -> Value:
    """x[..., i, j] * s[..., i]: s broadcast along the last dim."""
    rank = x.rank
    dims = _dims(rank)
    return b.elementwise([x, s], x.dtype, lambda r: r.mulf(r.args[0], r.args[1]),
                         maps=[identity_map(rank), map_of(rank, dims[:-1])])


def unary(b: FuncBuilder, x: Value, fn) -> Value:
    return b.elementwise([x], x.dtype, lambda r: fn(r, r.args[0]))


def silu(b: FuncBuilder, x: Value) -> Value:
    def body(r: Region) -> Value:
        one = r.const(1.0, "f32")
        e = r.exp(r.neg(r.args[0]))
        return r.divf(r.args[0], r.addf(one, e))
    return b.elementwise([x], x.dtype, body)


def silu_mul(b: FuncBuilder, gate: Value, up: Value) -> Value:
    def body(r: Region) -> Value:
        one = r.const(1.0, "f32")
        e = r.exp(r.neg(r.args[0]))
        return r.mulf(r.divf(r.args[0], r.addf(one, e)), r.args[1])
    return b.elementwise([gate, up], gate.dtype, body)


def sigmoid(b: FuncBuilder, x: Value) -> Value:
    def body(r: Region) -> Value:
        one = r.const(1.0, "f32")
        return r.divf(one, r.addf(one, r.exp(r.neg(r.args[0]))))
    return b.elementwise([x], x.dtype, body)


def relu(b: FuncBuilder, x: Value) -> Value:
    return b.elementwise([x], x.dtype, lambda r: r.maxf(r.args[0], r.const(0.0, x.dtype)))


def reduce_last(b: FuncBuilder, x: Value, kind: str) -> Value:
    """Reduce the last dim: sum, max, sumsq or absmax."""
    rank = x.rank
    dims = _dims(rank)
    shape = list(x.shape[:-1])
    init = {"sum": 0.0, "sumsq": 0.0, "max": float("-inf"), "absmax": 0.0}[kind]
    acc = b.fill(init, shape, x.dtype)

    def body(r: Region) -> list[Value]:
        v, a = r.args
        if kind == "sum":
            return [r.addf(v, a)]
        if kind == "sumsq":
            return [r.addf(r.mulf(v, v), a)]
        if kind == "max":
            return [r.maxf(v, a)]
        return [r.maxf(r.absf(v), a)]

    return b.generic([x], [acc], [identity_map(rank), map_of(rank, dims[:-1])],
                     ["parallel"] * (rank - 1) + ["reduction"], body)[0]


def softmax_last(b: FuncBuilder, x: Value) -> Value:
    rank = x.rank
    dims = _dims(rank)
    row = map_of(rank, dims[:-1])
    m = reduce_last(b, x, "max")
    e = b.elementwise([x, m], x.dtype, lambda r: r.exp(r.subf(r.args[0], r.args[1])),
                      maps=[identity_map(rank), row])
    s = reduce_last(b, e, "sum")
    return b.elementwise([e, s], x.dtype, lambda r: r.divf(r.args[0], r.args[1]),
                         maps=[identity_map(rank), row])


def rmsnorm(b: FuncBuilder, x: Value, weight: Value, eps: float, add_unit_offset: bool = False) -> Value:
    """x / sqrt(mean(x^2) + eps) * w over the last dim (w broadcast)."""
    rank = x.rank
    dims = _dims(rank)
    ss = reduce_last(b, x, "sumsq")
    n = x.shape[-1]

    def inv(r: Region) -> Value:
        mean = r.mulf(r.args[0], r.const(1.0 / n, "f32"))
        return r.rsqrt(r.addf(mean, r.const(eps, "f32")))

    scale = b.elementwise([ss], x.dtype, inv)

    def body(r: Region) -> Value:
        w = r.args[2]
        if add_unit_offset:
            w = r.addf(w, r.const(1.0, "f32"))
        return r.mulf(r.mulf(r.args[0], r.args[1]), w)

    return b.elementwise([x, scale, weight], x.dtype, body,
                         maps=[identity_map(rank), map_of(rank, dims[:-1]), map_of(rank, dims[-weight.rank:])])


# -- NumPy references -------------------------------------------------------

def ref_silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def ref_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ref_softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def ref_rmsnorm(x: np.ndarray, w: np.ndarray, eps: float, add_unit_offset: bool = False) -> np.ndarray:
    scale = 1.0 / np.sqrt((x.astype(np.float32) ** 2).mean(-1, keepdims=True) + eps)
    return (x * scale * ((w + 1.0) if add_unit_offset else w)).astype(np.float32)
