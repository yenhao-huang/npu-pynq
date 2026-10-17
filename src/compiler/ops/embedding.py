"""Token embedding, rotary position embedding, and tied output weights.

Tied embedding: the output projection ``lm_head`` is an NPU weight stored as
B tiles (src/isa/layout.py) of the INT8 matrix ``E^T [hidden, vocab]`` with one
scale per vocabulary entry. The token lookup reads row ``v`` of ``E`` straight
out of that tiled layout, so a tied model stores its embedding once.
"""

from __future__ import annotations

import numpy as np

from src.compiler.mlir_builder import Region, Value, identity_map, map_of
from src.compiler.ops.context import Ctx
from src.compiler.ops.linear import QuantizedWeight
from src.isa import isa, layout


def tied_lookup(ctx: Ctx, tokens: Value, name: str, w: QuantizedWeight, limits: isa.Limits) -> Value:
    """``tokens`` [T] i32 tensor -> [T, hidden] f32 rows of the tied weight."""
    b = ctx.b
    hidden, vocab = w.k, w.n
    if w.bits == 32:
        # FP32 reference path: the [hidden, vocab] matrix is dense; gather a column.
        dense = ctx.weight(f"{name}.w", w.q)

        def column(r: Region) -> Value:
            tok = r.index_cast(r.extract(tokens, [r.index(0)]), "index")
            return r.extract(dense, [r.index(1), tok])

        out = b.empty([tokens.shape[0], hidden], "f32")
        return b.generic([], [out], [identity_map(2)], ["parallel", "parallel"], lambda r: [column(r)])[0]
    tk = w.group if w.bits == 4 else min(hidden, limits.max_k)
    plan = layout.plan_gemm(1, vocab, hidden, limits, tk)
    packed = w.bits == 4
    flat = ctx.weight(f"{name}.q", w.q, shape=[plan.b_bytes // 2 if packed else plan.b_bytes], dtype="i8")
    scales = ctx.weight(f"{name}.s", w.scale)
    # The exporter checks the weight really is stored with this tiling.
    ctx.graph.meta.setdefault("require_npu_layout", {})[f"{name}.q"] = [hidden, vocab, plan.tk]
    t_count = tokens.shape[0]
    tn, nt = plan.tn, plan.nt
    last_width = plan.n_i(nt - 1)

    def body(r: Region) -> Value:
        t = r.index(0)
        h = r.index(1)
        tok = r.index_cast(r.extract(tokens, [t]), "index")
        idx = lambda v: r.const(v, "index")  # noqa: E731

        def op(name: str, x: Value, y: Value) -> Value:
            return r.op(f"arith.{name} {x}, {y} : index", "index")

        # B tiles are column panels: offset = ni*panel + kc*align8(tk*tn) + hk*width + vn.
        kc, hk = op("divui", h, idx(plan.tk)), op("remui", h, idx(plan.tk))
        ni, vn = op("divui", tok, idx(tn)), op("remui", tok, idx(tn))
        width = r.select(r.cmpi("eq", ni, idx(nt - 1)), idx(last_width), idx(tn))
        off = op("muli", ni, idx(plan.b_panel))
        off = op("addi", off, op("muli", kc, idx(isa.align(plan.tk * tn))))
        off = op("addi", off, op("muli", hk, width))
        off = op("addi", off, vn)
        if packed:
            byte = r.extract(flat, [op("shrui", off, idx(1))])
            odd = r.cmpi("eq", op("andi", off, idx(1)), idx(1))
            lo = r.op(f"arith.shrsi {r.op(f'arith.shli {byte}, {r.const(4, 'i8')} : i8', 'i8')}, {r.const(4, 'i8')} : i8", "i8")
            hi = r.op(f"arith.shrsi {byte}, {r.const(4, 'i8')} : i8", "i8")
            q = r.select(odd, hi, lo)
            s = r.extract(scales, [op("divui", h, idx(w.group)), tok])
        else:
            q = r.extract(flat, [off])
            s = r.extract(scales, [tok])
        return r.mulf(r.sitofp(q), s)

    out = b.empty([t_count, hidden], "f32")
    return b.generic([], [out], [identity_map(2)], ["parallel", "parallel"], lambda r: [body(r)])[0]


def dense_lookup(ctx: Ctx, tokens: Value, name: str, w: QuantizedWeight) -> Value:
    """Untied embedding: rows of an INT8 [vocab, hidden] table with per-row scales."""
    b = ctx.b
    table = ctx.weight(f"{name}.rows", np.ascontiguousarray(w.q.T))
    scales = ctx.weight(f"{name}.s", w.scale)

    def body(r: Region) -> Value:
        tok = r.index_cast(r.extract(tokens, [r.index(0)]), "index")
        q = r.extract(table, [tok, r.index(1)])
        return r.mulf(r.sitofp(q) if w.bits != 32 else q, r.extract(scales, [tok]))

    out = b.empty([tokens.shape[0], w.k], "f32")
    return b.generic([], [out], [identity_map(2)], ["parallel", "parallel"], lambda r: [body(r)])[0]


def rope_tables(max_seq: int, rotary_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim))
    angles = np.arange(max_seq, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rope(ctx: Ctx, x: Value, heads: int, head_dim: int, pos: Value, cos: Value, sin: Value,
         rotary_dim: int | None = None) -> Value:
    """Rotate-half RoPE on x [T, heads*head_dim] at positions pos .. pos+T-1.

    Only the first ``rotary_dim`` dims of each head rotate (partial rotary)."""
    b = ctx.b
    T = x.shape[0]
    R = rotary_dim or head_dim
    half = R // 2
    x3 = b.reshape(x, [T, heads, head_dim])

    def body(r: Region) -> Value:
        t, h, i = r.index(0), r.index(1), r.index(2)
        xv = r.args[0]
        p = r.addi(pos, t)
        in_rot = r.cmpi("ult", i, r.const(R, "index"))
        first = r.cmpi("ult", i, r.const(half, "index"))
        # Clamp indices so the extracts stay in bounds when i >= R.
        j = r.op(f"arith.remui {i}, {r.const(half, 'index')} : index", "index")
        partner_i = r.select(first, r.addi(i, r.const(half, "index")), r.subi(i, r.const(half, "index")))
        partner_i = r.select(in_rot, partner_i, i)
        partner = r.extract(x3, [t, h, partner_i])
        c = r.extract(cos, [p, j])
        s = r.extract(sin, [p, j])
        sign = r.select(first, r.const(-1.0, "f32"), r.const(1.0, "f32"))
        rotated = r.addf(r.mulf(xv, c), r.mulf(r.mulf(sign, partner), s))
        return r.select(in_rot, rotated, xv)

    out = b.elementwise([x3], "f32", body)
    return b.reshape(out, [T, heads * head_dim])


def ref_rope(x: np.ndarray, heads: int, head_dim: int, pos: int, cos: np.ndarray, sin: np.ndarray,
             rotary_dim: int | None = None) -> np.ndarray:
    T = x.shape[0]
    R = rotary_dim or head_dim
    half = R // 2
    x3 = x.reshape(T, heads, head_dim).copy()
    c = cos[pos:pos + T][:, None, :]
    s = sin[pos:pos + T][:, None, :]
    a, bb = x3[..., :half].copy(), x3[..., half:R].copy()
    x3[..., :half] = a * c - bb * s
    x3[..., half:R] = bb * c + a * s
    return x3.reshape(T, heads * head_dim)
