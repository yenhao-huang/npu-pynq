"""Mixture of experts: softmax router, top-k selection, sparse expert dispatch.

The router runs on the CPU in FP32. Top-k is a rank test (expert e is chosen
for token t when fewer than k experts score higher, ties broken by index),
which linalg expresses as a reduction. Each expert's SwiGLU MLP sits in an
``scf.if`` that runs only when some token of the call routes to it, so its
NPU tasks execute only for selected experts (decode: k of E).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.compiler.mlir_builder import Region, Value, identity_map, map_of
from src.compiler.ops import core, linear as L
from src.compiler.ops.context import Ctx


def topk_weights(ctx: Ctx, probs: Value, k: int, normalize: bool) -> Value:
    """[T, E] probabilities -> [T, E] combine weights (0 outside the top k)."""
    b = ctx.b
    T, E = probs.shape
    count = b.fill(0.0, [T, E], "f32")

    def rank(r: Region) -> list[Value]:
        pe, po, acc = r.args
        e, o = r.index(1), r.index(2)
        higher = r.cmpf("ogt", po, pe)
        tie = r.op(f"arith.andi {r.cmpf('oeq', po, pe)}, {r.cmpi('ult', o, e)} : i1", "i1")
        beats = r.op(f"arith.ori {higher}, {tie} : i1", "i1")
        return [r.select(beats, r.addf(acc, r.const(1.0, "f32")), acc)]

    ranks = b.generic([probs, probs], [count], [map_of(3, ["d0", "d1"]), map_of(3, ["d0", "d2"]),
                                                map_of(3, ["d0", "d1"])],
                      ["parallel", "parallel", "reduction"], rank)[0]
    chosen = b.elementwise([probs, ranks], "f32",
                           lambda r: r.select(r.cmpf("olt", r.args[1], r.const(float(k), "f32")),
                                              r.args[0], r.const(0.0, "f32")))
    if not normalize:
        return chosen
    total = core.reduce_last(b, chosen, "sum")
    return b.elementwise([chosen, total], "f32", lambda r: r.divf(r.args[0], r.args[1]),
                         maps=[identity_map(2), map_of(2, ["d0"])])


def moe(ctx: Ctx, x: Value, name: str, router: np.ndarray, experts: Sequence[dict], k: int,
        normalize: bool, inter: int) -> Value:
    """x [T, H]; experts[e] = {"gate_up": QuantizedWeight, "down": QuantizedWeight}."""
    b = ctx.b
    T, H = x.shape
    E = len(experts)
    logits = L.linear(ctx, x, f"{name}.router", L.quantize_weight(router, 32))
    weights = topk_weights(ctx, core.softmax_last(b, logits), k, normalize)
    used = core.reduce_last(b, b.transpose(weights, [1, 0]), "sum")  # [E]
    acc = b.fill(0.0, [T, H], "f32")
    for e, ex in enumerate(experts):
        hit = b.op(f"tensor.extract {used}[{b.index(e)}] : {used.type}", "f32")
        cond = b.op(f"arith.cmpf ogt, {hit}, {b.const(0.0, 'f32')} : f32", "i1")
        prev = acc

        def run(e=e, ex=ex, prev=prev):
            gu = L.linear(ctx, x, f"{name}.e{e}.gate_up", ex["gate_up"])
            act = core.silu_mul(b, b.extract_slice(gu, [0, 0], [T, inter]),
                                b.extract_slice(gu, [0, inter], [T, inter]))
            y = L.linear(ctx, act, f"{name}.e{e}.down", ex["down"])
            w_e = b.extract_slice(weights, [0, e], [T, 1], result_shape=[T])
            return [b.elementwise([prev, y, w_e], "f32",
                                  lambda r: r.addf(r.args[0], r.mulf(r.args[1], r.args[2])),
                                  maps=[identity_map(2), identity_map(2), map_of(2, ["d0"])])]

        acc = ctx.if_(cond, [acc.type], run, lambda prev=prev: [prev])[0]
    return acc


def ref_moe(x: np.ndarray, router: np.ndarray, experts: Sequence[dict], k: int, normalize: bool,
            inter: int) -> np.ndarray:
    logits = x @ router.T
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        top = np.argsort(-p[t], kind="stable")[:k]
        w = p[t, top] / (p[t, top].sum() if normalize else 1.0)
        for e, we in zip(top, w):
            gu = L.ref_linear(x[t:t + 1], experts[e]["gate_up"])
            act = core.ref_silu(gu[:, :inter]) * gu[:, inter:]
            out[t] += we * L.ref_linear(act, experts[e]["down"])[0]
    return out
