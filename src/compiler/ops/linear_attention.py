"""Linear attention: the gated delta rule (Gated DeltaNet, as in Qwen3.5).

Per layer the model keeps two recurrent states instead of a KV cache:

``conv``  [layers, channels, K-1]   the last K-1 inputs of a causal depthwise conv
``rec``   [layers, Hv, dk, dv]      the delta-rule memory S

For each token t (after a causal conv + SiLU over the mixed q/k/v projection,
and L2-normalized q, k with q scaled by 1/sqrt(dk)):

    S   <- S * exp(g_t)                       (per value head decay)
    kv  <- S^T k_t
    S   <- S + k_t (beta_t (v_t - kv))^T      (delta rule)
    o_t <- S^T q_t

Cost is O(dk * dv) per token and head regardless of context length. The
kernel is stateful, so it is written at the memref level like attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.compiler.mlir_builder import Value, float_literal, memref_type
from src.compiler.ops.attention import _K
from src.compiler.ops.context import Ctx


@dataclass(frozen=True)
class DeltaConfig:
    tokens: int
    k_heads: int
    v_heads: int
    dk: int
    dv: int
    conv_kernel: int
    layers: int

    @property
    def key_dim(self) -> int:
        return self.k_heads * self.dk

    @property
    def value_dim(self) -> int:
        return self.v_heads * self.dv

    @property
    def channels(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def name(self) -> str:
        return (f"gdn_t{self.tokens}_hk{self.k_heads}_hv{self.v_heads}_k{self.dk}_v{self.dv}"
                f"_c{self.conv_kernel}_l{self.layers}")

    def state_shapes(self) -> dict[str, tuple[tuple[int, ...], str]]:
        return {"conv": ((self.layers, self.channels, self.conv_kernel - 1), "f32"),
                "rec": ((self.layers, self.v_heads, self.dk, self.dv), "f32")}


def kernel_text(c: DeltaConfig) -> str:
    T, C, K = c.tokens, c.channels, c.conv_kernel
    HK, HV, DK, DV = c.k_heads, c.v_heads, c.dk, c.dv
    f = "fastmath<fast>"
    xt = memref_type([T, C], "f32")
    wt = memref_type([C, K], "f32")
    gt = memref_type([T, HV], "f32")
    st = memref_type([c.layers, C, K - 1], "f32")
    rt = memref_type([c.layers, HV, DK, DV], "f32")
    ot = memref_type([T, HV, DV], "f32")
    yt = memref_type([T, C], "f32")
    w = _K()
    cst: dict = {}
    header: list[str] = []

    def const(key, text: str) -> str:
        # Constants live in the entry block so every loop may use them.
        if key not in cst:
            w.n += 1
            cst[key] = f"%t{w.n}"
            header.append(f"    {cst[key]} = {text}")
        return cst[key]

    def ci(v: int) -> str:
        return const(("i", v), f"arith.constant {v} : index")

    def cf(v: float) -> str:
        return const(("f", v), f"arith.constant {float_literal(v)} : f32")

    zero, one = cf(0.0), cf(1.0)
    y = w.v(f"memref.alloc() : {yt}")
    # 1. Causal depthwise conv over [state (K-1 inputs), x_0 .. x_{T-1}], then SiLU.
    w.open(f"scf.for %t = {ci(0)} to {ci(T)} step {ci(1)}")
    w.open(f"scf.for %c = {ci(0)} to {ci(C)} step {ci(1)}")
    acc = zero
    for j in range(K):
        # Window slot j holds input at time t - (K-1) + j.
        src_t = w.v(f"arith.addi %t, {ci(j)} : index")  # index into [state ++ x]
        from_x = w.v(f"arith.cmpi uge, {src_t}, {ci(K - 1)} : index")
        xi = w.v(f"arith.subi {src_t}, {ci(K - 1)} : index")
        xi_safe = w.v(f"arith.select {from_x}, {xi}, {ci(0)} : index")
        si_safe = w.v(f"arith.select {from_x}, {ci(0)}, {src_t} : index")
        xv = w.v(f"memref.load %x[{xi_safe}, %c] : {xt}")
        sv = w.v(f"memref.load %cs[%layer, %c, {si_safe}] : {st}")
        v = w.v(f"arith.select {from_x}, {xv}, {sv} : f32")
        wj = w.v(f"memref.load %cw[%c, {ci(j)}] : {wt}")
        acc = w.v(f"arith.addf {acc}, {w.v(f'arith.mulf {v}, {wj} {f} : f32')} {f} : f32")
    e = w.v(f"math.exp {w.v(f'arith.negf {acc} : f32')} {f} : f32")
    sil = w.v(f"arith.divf {acc}, {w.v(f'arith.addf {one}, {e} : f32')} : f32")
    w.raw(f"memref.store {sil}, {y}[%t, %c] : {yt}")
    w.close()
    w.close()
    # The new conv state is the last K-1 entries of [state ++ x].
    tmp = w.v(f"memref.alloc() : {memref_type([C, K - 1], 'f32')}")
    w.open(f"scf.for %c = {ci(0)} to {ci(C)} step {ci(1)}")
    for j in range(K - 1):
        src = T + j  # position in [state ++ x] of new slot j
        if src >= K - 1:
            v = w.v(f"memref.load %x[{ci(src - (K - 1))}, %c] : {xt}")
        else:
            v = w.v(f"memref.load %cs[%layer, %c, {ci(src)}] : {st}")
        w.raw(f"memref.store {v}, {tmp}[%c, {ci(j)}] : {memref_type([C, K - 1], 'f32')}")
    for j in range(K - 1):
        v = w.v(f"memref.load {tmp}[%c, {ci(j)}] : {memref_type([C, K - 1], 'f32')}")
        w.raw(f"memref.store {v}, %cs[%layer, %c, {ci(j)}] : {st}")
    w.close()

    # 2. Per token and value head: L2 norms, then the gated delta rule.
    q_scale = cf(1.0 / np.sqrt(DK))
    group = HV // HK
    kvm = w.v(f"memref.alloca() : memref<{DV}xf32>")
    w.open(f"scf.for %t = {ci(0)} to {ci(T)} step {ci(1)}")
    w.open(f"scf.for %h = {ci(0)} to {ci(HV)} step {ci(1)}")
    hk = w.v(f"arith.divui %h, {ci(group)} : index")
    qb = w.v(f"arith.muli {hk}, {ci(DK)} : index")
    kb = w.v(f"arith.addi {qb}, {ci(c.key_dim)} : index")
    vb = w.v(f"arith.addi {w.v(f'arith.muli %h, {ci(DV)} : index')}, {ci(2 * c.key_dim)} : index")

    def sumsq(base: str) -> str:
        r = w.v(f"scf.for %d = {ci(0)} to {ci(DK)} step {ci(1)} iter_args(%a = {zero}) -> (f32) {{")
        w.depth += 1
        x = w.v(f"memref.load {y}[%t, {w.v(f'arith.addi {base}, %d : index')}] : {yt}")
        w.raw(f"scf.yield {w.v(f'arith.addf %a, {w.v(f'arith.mulf {x}, {x} {f} : f32')} {f} : f32')} : f32")
        w.close()
        return w.v(f"math.rsqrt {w.v(f'arith.addf {r}, {cf(1e-6)} : f32')} : f32")

    qn = w.v(f"arith.mulf {sumsq(qb)}, {q_scale} : f32")
    kn = sumsq(kb)
    decay = w.v(f"math.exp {w.v(f'memref.load %g[%t, %h] : {gt}')} : f32")
    beta = w.v(f"memref.load %beta[%t, %h] : {gt}")
    # S <- S * decay; kv = S^T k
    w.open(f"scf.for %j = {ci(0)} to {ci(DV)} step {ci(1)}")
    w.raw(f"memref.store {zero}, {kvm}[%j] : memref<{DV}xf32>")
    w.close()
    w.open(f"scf.for %i = {ci(0)} to {ci(DK)} step {ci(1)}")
    ki = w.v(f"arith.mulf {w.v(f'memref.load {y}[%t, {w.v(f'arith.addi {kb}, %i : index')}] : {yt}')}, {kn} : f32")
    w.open(f"scf.for %j = {ci(0)} to {ci(DV)} step {ci(1)}")
    s = w.v(f"arith.mulf {w.v(f'memref.load %rs[%layer, %h, %i, %j] : {rt}')}, {decay} {f} : f32")
    w.raw(f"memref.store {s}, %rs[%layer, %h, %i, %j] : {rt}")
    a = w.v(f"memref.load {kvm}[%j] : memref<{DV}xf32>")
    w.raw(f"memref.store {w.v(f'arith.addf {a}, {w.v(f'arith.mulf {s}, {ki} {f} : f32')} {f} : f32')}, {kvm}[%j] : memref<{DV}xf32>")
    w.close()
    w.close()
    # delta = beta (v - kv)  (stored back into kvm)
    w.open(f"scf.for %j = {ci(0)} to {ci(DV)} step {ci(1)}")
    v = w.v(f"memref.load {y}[%t, {w.v(f'arith.addi {vb}, %j : index')}] : {yt}")
    d = w.v(f"arith.mulf {w.v(f'arith.subf {v}, {w.v(f'memref.load {kvm}[%j] : memref<{DV}xf32>')} : f32')}, {beta} : f32")
    w.raw(f"memref.store {d}, {kvm}[%j] : memref<{DV}xf32>")
    w.raw(f"memref.store {zero}, %out[%t, %h, %j] : {ot}")
    w.close()
    # S += k delta^T; out = S^T q
    w.open(f"scf.for %i = {ci(0)} to {ci(DK)} step {ci(1)}")
    ki = w.v(f"arith.mulf {w.v(f'memref.load {y}[%t, {w.v(f'arith.addi {kb}, %i : index')}] : {yt}')}, {kn} : f32")
    qi = w.v(f"arith.mulf {w.v(f'memref.load {y}[%t, {w.v(f'arith.addi {qb}, %i : index')}] : {yt}')}, {qn} : f32")
    w.open(f"scf.for %j = {ci(0)} to {ci(DV)} step {ci(1)}")
    dj = w.v(f"memref.load {kvm}[%j] : memref<{DV}xf32>")
    s = w.v(f"arith.addf {w.v(f'memref.load %rs[%layer, %h, %i, %j] : {rt}')}, {w.v(f'arith.mulf {ki}, {dj} {f} : f32')} {f} : f32")
    w.raw(f"memref.store {s}, %rs[%layer, %h, %i, %j] : {rt}")
    o = w.v(f"memref.load %out[%t, %h, %j] : {ot}")
    w.raw(f"memref.store {w.v(f'arith.addf {o}, {w.v(f'arith.mulf {s}, {qi} {f} : f32')} {f} : f32')}, %out[%t, %h, %j] : {ot}")
    w.close()
    w.close()
    w.close()
    w.close()
    w.raw("return")
    args = [f"%x: {xt}", f"%cw: {wt}", f"%g: {gt}", f"%beta: {gt}", f"%cs: {st}", f"%rs: {rt}",
            f"%out: {ot}", "%layer: index"]
    return f"  func.func private @{c.name}({', '.join(args)}) {{\n" + "\n".join(header + w.lines) + "\n  }\n"


def gated_delta(ctx: Ctx, cfg: DeltaConfig, mixed: Value, conv_w: Value, g: Value, beta: Value,
                states: dict[str, Value], layer: Value) -> Value:
    """mixed [T, channels] (pre-conv q|k|v), g/beta [T, Hv] -> [T, Hv*dv]; updates the states."""
    b = ctx.b
    key = ("kernel", cfg.name)
    if key not in ctx.cache:
        ctx.graph.module.add_raw(kernel_text(cfg))
        ctx.cache[key] = True

    def buf(t: Value) -> Value:
        mt = t.type.replace("tensor<", "memref<")
        return b.op(f"bufferization.to_buffer {t} read_only : {t.type} to {mt}", mt)

    T = cfg.tokens
    out_t = memref_type([T, cfg.v_heads, cfg.dv], "f32")
    out = b.op(f"memref.alloc() : {out_t}", out_t)
    b.call(cfg.name, [buf(mixed), buf(conv_w), buf(g), buf(beta), states["conv"], states["rec"], out, layer])
    o = b.op(f"bufferization.to_tensor {out} restrict writable : {out_t} to tensor<{T}x{cfg.v_heads}x{cfg.dv}xf32>",
             f"tensor<{T}x{cfg.v_heads}x{cfg.dv}xf32>")
    return o


def ref_gated_delta(cfg: DeltaConfig, mixed, conv_w, g, beta, state: dict, layer: int) -> np.ndarray:
    T, C, K = cfg.tokens, cfg.channels, cfg.conv_kernel
    seq = np.concatenate([state["conv"][layer].T, mixed], 0)  # [K-1+T, C]
    y = np.stack([(seq[t:t + K] * conv_w.T).sum(0) for t in range(T)])
    y = y / (1 + np.exp(-y))
    state["conv"][layer] = seq[-(K - 1):].T
    kd = cfg.key_dim
    q = y[:, :kd].reshape(T, cfg.k_heads, cfg.dk)
    k = y[:, kd:2 * kd].reshape(T, cfg.k_heads, cfg.dk)
    v = y[:, 2 * kd:].reshape(T, cfg.v_heads, cfg.dv)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / np.sqrt(cfg.dk)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    rep = cfg.v_heads // cfg.k_heads
    q, k = np.repeat(q, rep, 1), np.repeat(k, rep, 1)
    S = state["rec"][layer]
    out = np.zeros((T, cfg.v_heads, cfg.dv), np.float32)
    for t in range(T):
        S *= np.exp(g[t])[:, None, None]
        kv = np.einsum("hij,hi->hj", S, k[t])
        delta = (v[t] - kv) * beta[t][:, None]
        S += k[t][:, :, None] * delta[:, None, :]
        out[t] = np.einsum("hij,hi->hj", S, q[t])
    return out
