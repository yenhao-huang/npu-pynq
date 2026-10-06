"""Causal attention over a KV cache, compiled for the CPU.

The kernel is stateful (it appends this step's keys and values to the cache,
then attends over every cached position), so it is written at the memref
level with scf loops; LLVM vectorizes the inner loops over the head
dimension. Variants:

``mode="naive"``   scores for all positions, then softmax, then P @ V
``mode="online"``  single pass with a running max and sum (flash-style)
``kv="f32"``       cache holds FP32 keys/values
``kv="int8"``      cache holds INT8 keys/values with one FP32 scale per
                   (layer, kv head, position): KV-cache quantization

GQA/MQA: query head ``h`` reads kv head ``h // (heads / kv_heads)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.compiler.mlir_builder import Value, float_literal, memref_type
from src.compiler.ops.context import Ctx


@dataclass(frozen=True)
class AttnConfig:
    tokens: int        # queries per call (1 for decode, the chunk for prefill)
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    max_seq: int
    mode: str = "online"
    kv: str = "f32"
    scale: float | None = None

    @property
    def name(self) -> str:
        return (f"attn_t{self.tokens}_h{self.heads}_kv{self.kv_heads}_d{self.head_dim}"
                f"_l{self.layers}_s{self.max_seq}_{self.mode}_{self.kv}")

    def cache_shapes(self) -> dict[str, tuple[tuple[int, ...], str]]:
        data = (self.layers, self.kv_heads, self.max_seq, self.head_dim)
        if self.kv == "int8":
            scales = (self.layers, self.kv_heads, self.max_seq)
            return {"k": (data, "i8"), "v": (data, "i8"), "ks": (scales, "f32"), "vs": (scales, "f32")}
        return {"k": (data, "f32"), "v": (data, "f32")}


class _K:
    """Minimal SSA text writer for hand-written kernels."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.n = 0
        self.depth = 2

    def v(self, text: str) -> str:
        self.n += 1
        name = f"%t{self.n}"
        self.lines.append("  " * self.depth + f"{name} = {text}")
        return name

    def vn(self, count: int, text: str) -> str:
        self.n += 1
        name = f"%t{self.n}"
        self.lines.append("  " * self.depth + f"{name}:{count} = {text}")
        return name

    def raw(self, text: str) -> None:
        self.lines.append("  " * self.depth + text)

    def open(self, text: str) -> None:
        self.raw(text + " {")
        self.depth += 1

    def close(self, text: str = "}") -> None:
        self.depth -= 1
        self.raw(text)


def kernel_text(c: AttnConfig) -> str:
    T, H, HK, D, L, S = c.tokens, c.heads, c.kv_heads, c.head_dim, c.layers, c.max_seq
    group = H // HK
    scale = c.scale if c.scale is not None else 1.0 / np.sqrt(D)
    f = "fastmath<fast>"
    qt = memref_type([T, H, D], "f32")
    kt = memref_type([T, HK, D], "f32")
    shapes = c.cache_shapes()
    ct = memref_type(shapes["k"][0], shapes["k"][1])
    st = memref_type(shapes["ks"][0], "f32") if c.kv == "int8" else ""
    args = [f"%q: {qt}", f"%k: {kt}", f"%v: {kt}", f"%kc: {ct}", f"%vc: {ct}"]
    if c.kv == "int8":
        args += [f"%ksc: {st}", f"%vsc: {st}"]
    args += [f"%out: {qt}", "%layer: index", "%pos: index"]
    w = _K()
    i0 = w.v("arith.constant 0 : index")
    i1 = w.v("arith.constant 1 : index")
    iT = w.v(f"arith.constant {T} : index")
    iH = w.v(f"arith.constant {H} : index")
    iHK = w.v(f"arith.constant {HK} : index")
    iD = w.v(f"arith.constant {D} : index")
    iS = w.v(f"arith.constant {S} : index")
    iG = w.v(f"arith.constant {group} : index")
    zero = w.v("arith.constant 0.0 : f32")
    one = w.v("arith.constant 1.0 : f32")
    ninf = w.v("arith.constant 0xFF800000 : f32")
    fscale = w.v(f"arith.constant {float_literal(scale)} : f32")

    # 1. Append this call's keys/values at positions pos .. pos+T-1.
    w.open(f"scf.for %t = {i0} to {iT} step {i1}")
    p = w.v(f"arith.addi %pos, %t : index")
    inb = w.v(f"arith.cmpi ult, {p}, {iS} : index")
    w.open(f"scf.if {inb}")
    w.open(f"scf.for %h = {i0} to {iHK} step {i1}")
    for src, dst, sdst in (("%k", "%kc", "%ksc"), ("%v", "%vc", "%vsc")):
        if c.kv == "f32":
            w.open(f"scf.for %d = {i0} to {iD} step {i1}")
            x = w.v(f"memref.load {src}[%t, %h, %d] : {kt}")
            w.raw(f"memref.store {x}, {dst}[%layer, %h, {p}, %d] : {ct}")
            w.close()
        else:
            amax = w.v(f"scf.for %d = {i0} to {iD} step {i1} iter_args(%m = {zero}) -> (f32) {{")
            w.depth += 1
            x = w.v(f"memref.load {src}[%t, %h, %d] : {kt}")
            a = w.v(f"math.absf {x} : f32")
            m2 = w.v(f"arith.maximumf %m, {a} : f32")
            w.raw(f"scf.yield {m2} : f32")
            w.close()
            eps = w.v(f"arith.constant {float_literal(1e-8)} : f32")
            amax2 = w.v(f"arith.maximumf {amax}, {eps} : f32")
            sc = w.v(f"arith.divf {amax2}, {w.v('arith.constant 127.0 : f32')} : f32")
            inv = w.v(f"arith.divf {one}, {sc} : f32")
            w.raw(f"memref.store {sc}, {sdst}[%layer, %h, {p}] : {st}")
            w.open(f"scf.for %d = {i0} to {iD} step {i1}")
            x = w.v(f"memref.load {src}[%t, %h, %d] : {kt}")
            y = w.v(f"arith.mulf {x}, {inv} : f32")
            r = w.v(f"math.roundeven {y} : f32")
            qi = w.v(f"arith.fptosi {r} : f32 to i8")
            w.raw(f"memref.store {qi}, {dst}[%layer, %h, {p}, %d] : {ct}")
            w.close()
    w.close()
    w.close()
    w.close()

    def load_kv(cache: str, scales: str, j: str, h: str, d: str) -> str:
        if c.kv == "f32":
            return w.v(f"memref.load {cache}[%layer, {h}, {j}, {d}] : {ct}")
        qv = w.v(f"memref.load {cache}[%layer, {h}, {j}, {d}] : {ct}")
        return w.v(f"arith.sitofp {qv} : i8 to f32")

    def kv_scale(scales: str, j: str, h: str) -> str | None:
        if c.kv == "f32":
            return None
        return w.v(f"memref.load {scales}[%layer, {h}, {j}] : {st}")

    def dot(t: str, hq: str, h: str, j: str) -> str:
        s = w.v(f"scf.for %d = {i0} to {iD} step {i1} iter_args(%acc = {zero}) -> (f32) {{")
        w.depth += 1
        qv = w.v(f"memref.load %q[{t}, {hq}, %d] : {qt}")
        kv = load_kv("%kc", "%ksc", j, h, "%d")
        prod = w.v(f"arith.mulf {qv}, {kv} {f} : f32")
        acc2 = w.v(f"arith.addf %acc, {prod} {f} : f32")
        w.raw(f"scf.yield {acc2} : f32")
        w.close()
        ks = kv_scale("%ksc", j, h)
        if ks:
            s = w.v(f"arith.mulf {s}, {ks} {f} : f32")
        return w.v(f"arith.mulf {s}, {fscale} {f} : f32")

    # 2. Causal attention: query t sees positions 0 .. pos+t.
    if c.mode == "naive":
        scores = w.v(f"memref.alloca() : memref<{S}xf32>")
    w.open(f"scf.for %t = {i0} to {iT} step {i1}")
    p = w.v("arith.addi %pos, %t : index")
    p1 = w.v(f"arith.addi {p}, {i1} : index")
    n = w.v(f"arith.minui {p1}, {iS} : index")
    w.open(f"scf.for %hq = {i0} to {iH} step {i1}")
    h = w.v(f"arith.divui %hq, {iG} : index")
    w.open(f"scf.for %d = {i0} to {iD} step {i1}")
    w.raw(f"memref.store {zero}, %out[%t, %hq, %d] : {qt}")
    w.close()
    if c.mode == "naive":
        mx = w.v(f"scf.for %j = {i0} to {n} step {i1} iter_args(%m = {ninf}) -> (f32) {{")
        w.depth += 1
        s = dot("%t", "%hq", h, "%j")
        w.raw(f"memref.store {s}, {scores}[%j] : memref<{S}xf32>")
        m2 = w.v(f"arith.maximumf %m, {s} : f32")
        w.raw(f"scf.yield {m2} : f32")
        w.close()
        tot = w.v(f"scf.for %j = {i0} to {n} step {i1} iter_args(%l = {zero}) -> (f32) {{")
        w.depth += 1
        s = w.v(f"memref.load {scores}[%j] : memref<{S}xf32>")
        e = w.v(f"math.exp {w.v(f'arith.subf {s}, {mx} : f32')} {f} : f32")
        w.raw(f"memref.store {e}, {scores}[%j] : memref<{S}xf32>")
        l2 = w.v(f"arith.addf %l, {e} {f} : f32")
        w.raw(f"scf.yield {l2} : f32")
        w.close()
        inv = w.v(f"arith.divf {one}, {tot} : f32")
        w.open(f"scf.for %j = {i0} to {n} step {i1}")
        pj = w.v(f"arith.mulf {w.v(f'memref.load {scores}[%j] : memref<{S}xf32>')}, {inv} {f} : f32")
        vs = kv_scale("%vsc", "%j", h)
        if vs:
            pj = w.v(f"arith.mulf {pj}, {vs} {f} : f32")
        w.open(f"scf.for %d = {i0} to {iD} step {i1}")
        vv = load_kv("%vc", "%vsc", "%j", h, "%d")
        o = w.v(f"memref.load %out[%t, %hq, %d] : {qt}")
        o2 = w.v(f"arith.addf {o}, {w.v(f'arith.mulf {pj}, {vv} {f} : f32')} {f} : f32")
        w.raw(f"memref.store {o2}, %out[%t, %hq, %d] : {qt}")
        w.close()
        w.close()
    else:
        res = w.vn(2, f"scf.for %j = {i0} to {n} step {i1} iter_args(%m = {ninf}, %l = {zero}) -> (f32, f32) {{")
        w.depth += 1
        s = dot("%t", "%hq", h, "%j")
        m2 = w.v(f"arith.maximumf %m, {s} : f32")
        corr = w.v(f"math.exp {w.v(f'arith.subf %m, {m2} : f32')} {f} : f32")
        pj = w.v(f"math.exp {w.v(f'arith.subf {s}, {m2} : f32')} {f} : f32")
        l2 = w.v(f"arith.addf {w.v(f'arith.mulf %l, {corr} {f} : f32')}, {pj} {f} : f32")
        vs = kv_scale("%vsc", "%j", h)
        pv = w.v(f"arith.mulf {pj}, {vs} {f} : f32") if vs else pj
        w.open(f"scf.for %d = {i0} to {iD} step {i1}")
        vv = load_kv("%vc", "%vsc", "%j", h, "%d")
        o = w.v(f"memref.load %out[%t, %hq, %d] : {qt}")
        o2 = w.v(f"arith.addf {w.v(f'arith.mulf {o}, {corr} {f} : f32')}, "
                 f"{w.v(f'arith.mulf {pv}, {vv} {f} : f32')} {f} : f32")
        w.raw(f"memref.store {o2}, %out[%t, %hq, %d] : {qt}")
        w.close()
        w.raw(f"scf.yield {m2}, {l2} : f32, f32")
        w.close()
        inv = w.v(f"arith.divf {one}, {res}#1 : f32")
        w.open(f"scf.for %d = {i0} to {iD} step {i1}")
        o = w.v(f"memref.load %out[%t, %hq, %d] : {qt}")
        w.raw(f"memref.store {w.v(f'arith.mulf {o}, {inv} {f} : f32')}, %out[%t, %hq, %d] : {qt}")
        w.close()
    w.close()
    w.close()
    w.raw("return")
    body = "\n".join(w.lines)
    return f"  func.func private @{c.name}({', '.join(args)}) {{\n{body}\n  }}\n"


def attention(ctx: Ctx, cfg: AttnConfig, q: Value, k: Value, v: Value, caches: dict[str, Value],
              layer: Value, pos: Value) -> Value:
    """q [T, H*D], k/v [T, HK*D] tensors -> [T, H*D]; updates the caches."""
    b = ctx.b
    key = ("kernel", cfg.name)
    if key not in ctx.cache:
        ctx.graph.module.add_raw(kernel_text(cfg))
        ctx.cache[key] = True
    T, H, HK, D = cfg.tokens, cfg.heads, cfg.kv_heads, cfg.head_dim

    def buf(t: Value, heads: int) -> Value:
        t3 = b.reshape(t, [T, heads, D])
        mt = memref_type([T, heads, D], "f32")
        return b.op(f"bufferization.to_buffer {t3} read_only : {t3.type} to {mt}", mt)

    qb, kb, vb = buf(q, H), buf(k, HK), buf(v, HK)
    out_t = memref_type([T, H, D], "f32")
    out = b.op(f"memref.alloc() : {out_t}", out_t)
    args = [qb, kb, vb, caches["k"], caches["v"]]
    if cfg.kv == "int8":
        args += [caches["ks"], caches["vs"]]
    b.call(cfg.name, [*args, out, layer, pos])
    o = b.op(f"bufferization.to_tensor {out} restrict writable : {out_t} to tensor<{T}x{H}x{D}xf32>",
             f"tensor<{T}x{H}x{D}xf32>")
    return b.reshape(o, [T, H * D])


def ref_attention(cfg: AttnConfig, q, k, v, cache: dict, layer: int, pos: int) -> np.ndarray:
    """NumPy model of the kernel, including INT8 cache rounding."""
    T, H, HK, D, S = cfg.tokens, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.max_seq
    q = q.reshape(T, H, D)
    k = k.reshape(T, HK, D)
    v = v.reshape(T, HK, D)
    for t in range(T):
        p = pos + t
        if p >= S:
            continue
        for name, src in (("k", k), ("v", v)):
            if cfg.kv == "f32":
                cache[name][layer, :, p] = src[t]
            else:
                sc = np.maximum(np.abs(src[t]).max(-1), np.float32(1e-8)) / np.float32(127.0)
                cache[name + "s"][layer, :, p] = sc
                cache[name][layer, :, p] = np.round(src[t] * (1.0 / sc)[:, None]).astype(np.int8)
    scale = cfg.scale if cfg.scale is not None else 1.0 / np.sqrt(D)
    out = np.zeros((T, H, D), np.float32)
    for t in range(T):
        n = min(pos + t + 1, S)
        for hq in range(H):
            h = hq // (H // HK)
            kk = cache["k"][layer, h, :n].astype(np.float32)
            vv = cache["v"][layer, h, :n].astype(np.float32)
            if cfg.kv == "int8":
                kk = kk * cache["ks"][layer, h, :n, None]
                vv = vv * cache["vs"][layer, h, :n, None]
            s = kk @ q[t, hq] * scale
            pr = np.exp(s - s.max())
            out[t, hq] = (pr / pr.sum()) @ vv
    return out.reshape(T, H * D)
