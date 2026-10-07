"""Qwen3 dense: tied embedding, RMSNorm, per-head q/k RMSNorm, RoPE, GQA, SwiGLU.

Built only from src/compiler/ops; differs from SmolLM in the q/k norms, an
explicit head_dim (heads * head_dim != hidden) and bias-free projections.
"""

from __future__ import annotations

import json

import numpy as np

from src.compiler.ops import attention as A, core, embedding as E, linear as L
from src.inference.graph import LLMOptions, StateSpec, build_entry_points, choose_group, new_graph
from src.inference.weights import SafeTensors, resolve_hf

DEFAULT_SOURCE = "Qwen/Qwen3-0.6B"


class Config:
    def __init__(self, hf: dict) -> None:
        self.hidden = hf["hidden_size"]
        self.inter = hf["intermediate_size"]
        self.heads = hf["num_attention_heads"]
        self.kv_heads = hf["num_key_value_heads"]
        self.head_dim = hf.get("head_dim") or self.hidden // self.heads
        self.layers = hf["num_hidden_layers"]
        self.eps = hf["rms_norm_eps"]
        self.theta = hf.get("rope_theta", 1000000.0)
        self.vocab = hf["vocab_size"]
        self.eos = hf.get("eos_token_id")


def layer_weights(st: SafeTensors, cfg: Config, layer: int, opts: LLMOptions, awq=None) -> dict:
    p = f"model.layers.{layer}."
    w = lambda n: st[p + n]  # noqa: E731
    mats = {
        "qkv": np.concatenate([w("self_attn.q_proj.weight"), w("self_attn.k_proj.weight"),
                               w("self_attn.v_proj.weight")], axis=0),
        "o": w("self_attn.o_proj.weight"),
        "gate_up": np.concatenate([w("mlp.gate_proj.weight"), w("mlp.up_proj.weight")], axis=0),
        "down": w("mlp.down_proj.weight"),
    }
    norms = {"ln1": w("input_layernorm.weight"), "ln2": w("post_attention_layernorm.weight")}
    if awq is not None:
        mats, norms = awq(layer, mats, norms)
    out = {k: L.quantize_weight(v, opts.bits, opts.group) for k, v in mats.items()}
    out.update({k: v.astype(np.float32) for k, v in norms.items()})
    out["q_norm"] = w("self_attn.q_norm.weight").astype(np.float32)
    out["k_norm"] = w("self_attn.k_norm.weight").astype(np.float32)
    return out


def awq_spec(cfg: Config):
    from src.inference.quant.awq import FoldSpec

    layer = "model.layers.{i}."
    return FoldSpec(layer + "self_attn.q_proj", layer + "self_attn.o_proj", layer + "mlp.gate_proj",
                    layer + "mlp.down_proj", cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim,
                    cfg.inter, cfg.heads, cfg.kv_heads, cfg.head_dim)


def build(source: str = DEFAULT_SOURCE, opts: LLMOptions | None = None, awq=None):
    opts = opts or LLMOptions()
    root = resolve_hf(source)
    cfg = Config(json.loads((root / "config.json").read_text()))
    if opts.layers:
        cfg.layers = opts.layers
    st = SafeTensors(root)
    graph = new_graph("qwen3")
    H, HK, D = cfg.heads, cfg.kv_heads, cfg.head_dim
    emb = L.quantize_weight(st["model.embed_tokens.weight"], opts.embed_bits, opts.group)
    cos, sin = E.rope_tables(opts.max_seq, D, cfg.theta)
    opts.group = choose_group(opts.group, [cfg.hidden, cfg.inter, cfg.heads * cfg.head_dim])
    if opts.awq and awq is None:
        from src.inference.quant import awq as AWQ
        awq = AWQ.hook_for(root, AWQ.calibration_ids(root), awq_spec(cfg), cfg.layers, opts.bits, opts.group)
    layers = [layer_weights(st, cfg, i, opts, awq) for i in range(cfg.layers)]
    final_norm = st["model.norm.weight"].astype(np.float32)
    kv_shapes = A.AttnConfig(1, H, HK, D, cfg.layers, opts.max_seq, opts.attention, opts.kv).cache_shapes()
    state = [StateSpec(n, s, d) for n, (s, d) in kv_shapes.items()]

    def head_norm(ctx, x, heads, weight):
        b = ctx.b
        T = x.shape[0]
        y = core.rmsnorm(b, b.reshape(x, [T, heads, D]), weight, cfg.eps)
        return b.reshape(y, [T, heads * D])

    def body(ctx, tokens, st_vals, pos, last):
        b = ctx.b
        T = tokens.shape[0]
        attn = A.AttnConfig(T, H, HK, D, cfg.layers, opts.max_seq, opts.attention, opts.kv)
        cos_v, sin_v = ctx.weight("rope.cos", cos), ctx.weight("rope.sin", sin)
        x = E.tied_lookup(ctx, tokens, "lm_head", emb, opts.limits)
        q_dim, kv_dim = H * D, HK * D
        for i, lw in enumerate(layers):
            h = core.rmsnorm(b, x, ctx.weight(f"l{i}.ln1", lw["ln1"]), cfg.eps)
            qkv = L.linear(ctx, h, f"l{i}.qkv", lw["qkv"])
            q = head_norm(ctx, b.extract_slice(qkv, [0, 0], [T, q_dim]), H, ctx.weight(f"l{i}.q_norm", lw["q_norm"]))
            k = head_norm(ctx, b.extract_slice(qkv, [0, q_dim], [T, kv_dim]), HK,
                          ctx.weight(f"l{i}.k_norm", lw["k_norm"]))
            v = b.extract_slice(qkv, [0, q_dim + kv_dim], [T, kv_dim])
            q = E.rope(ctx, q, H, D, pos, cos_v, sin_v)
            k = E.rope(ctx, k, HK, D, pos, cos_v, sin_v)
            o = A.attention(ctx, attn, q, k, v, st_vals, b.index(i), pos)
            x = core.add(b, x, L.linear(ctx, o, f"l{i}.o", lw["o"]))
            h = core.rmsnorm(b, x, ctx.weight(f"l{i}.ln2", lw["ln2"]), cfg.eps)
            gu = L.linear(ctx, h, f"l{i}.gate_up", lw["gate_up"])
            act = core.silu_mul(b, b.extract_slice(gu, [0, 0], [T, cfg.inter]),
                                b.extract_slice(gu, [0, cfg.inter], [T, cfg.inter]))
            x = core.add(b, x, L.linear(ctx, act, f"l{i}.down", lw["down"]))
        x = core.rmsnorm(b, x, ctx.weight("norm", final_norm), cfg.eps)
        if last is not None:
            x = b.extract_slice(x, [last, 0], [1, cfg.hidden])
        return L.linear(ctx, x, "lm_head", emb)

    build_entry_points(graph, state, cfg.vocab, opts.chunk, body)
    graph.meta.update({"family": "qwen3", "config": vars(cfg), "options": opts.describe(),
                       "eos": cfg.eos, "source": source})
    graph.files["tokenizer.json"] = root / "tokenizer.json"
    return graph, cfg
