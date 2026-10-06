"""Qwen3-MoE: Qwen3 attention with sparse mixture-of-experts MLPs.

Router (FP32, CPU), top-k selection and the weighted combine are CPU code;
each expert's gate/up and down projections are NPU tasks guarded by an
scf.if, so only routed experts run. Layers listed in mlp_only_layers (or
off the sparse step) keep a dense SwiGLU MLP.
"""

from __future__ import annotations

import json

import numpy as np

from src.compiler.ops import attention as A, core, embedding as E, linear as L, moe as MOE
from src.inference.graph import LLMOptions, StateSpec, build_entry_points, choose_group, new_graph
from src.inference.weights import SafeTensors, resolve_hf

DEFAULT_SOURCE = "build/tiny-qwen3-moe"  # exp/1007_sw_stack/host/make_tiny_moe.py


class Config:
    def __init__(self, hf: dict) -> None:
        self.hidden = hf["hidden_size"]
        self.inter = hf["intermediate_size"]
        self.moe_inter = hf["moe_intermediate_size"]
        self.experts = hf.get("num_experts", hf.get("num_local_experts", 0))
        self.top_k = hf["num_experts_per_tok"]
        self.norm_topk = hf.get("norm_topk_prob", False)
        self.sparse_step = hf.get("decoder_sparse_step", 1)
        self.mlp_only = set(hf.get("mlp_only_layers") or [])
        self.heads = hf["num_attention_heads"]
        self.kv_heads = hf["num_key_value_heads"]
        self.head_dim = hf.get("head_dim") or self.hidden // self.heads
        self.layers = hf["num_hidden_layers"]
        self.eps = hf["rms_norm_eps"]
        self.theta = hf.get("rope_theta") or (hf.get("rope_parameters") or {}).get("rope_theta", 1000000.0)
        self.vocab = hf["vocab_size"]
        self.tied = hf.get("tie_word_embeddings", False)
        self.eos = hf.get("eos_token_id")

    def sparse(self, layer: int) -> bool:
        return layer not in self.mlp_only and self.experts > 0 and (layer + 1) % self.sparse_step == 0


def expert_weights(st: SafeTensors, p: str, cfg: Config):
    if p + "mlp.experts.gate_up_proj" in st:
        gu, down = st[p + "mlp.experts.gate_up_proj"], st[p + "mlp.experts.down_proj"]
        return [(gu[e], down[e]) for e in range(cfg.experts)]
    out = []
    for e in range(cfg.experts):
        q = f"{p}mlp.experts.{e}."
        out.append((np.concatenate([st[q + "gate_proj.weight"], st[q + "up_proj.weight"]]), st[q + "down_proj.weight"]))
    return out


def build(source: str = DEFAULT_SOURCE, opts: LLMOptions | None = None, awq=None):
    opts = opts or LLMOptions()
    root = resolve_hf(source)
    cfg = Config(json.loads((root / "config.json").read_text()))
    if opts.layers:
        cfg.layers = opts.layers
    opts.group = choose_group(opts.group, [cfg.hidden, cfg.moe_inter, cfg.inter, cfg.heads * cfg.head_dim])
    st = SafeTensors(root)
    graph = new_graph("qwen3_moe")
    H, HK, D = cfg.heads, cfg.kv_heads, cfg.head_dim
    q = lambda w: L.quantize_weight(w, opts.bits, opts.group)  # noqa: E731
    embed = L.quantize_weight(st["model.embed_tokens.weight"], opts.embed_bits, opts.group)
    head = embed if cfg.tied else L.quantize_weight(st["lm_head.weight"], opts.embed_bits, opts.group)
    cos, sin = E.rope_tables(opts.max_seq, D, cfg.theta)
    layers = []
    for i in range(cfg.layers):
        p = f"model.layers.{i}."
        lw = {"qkv": q(np.concatenate([st[p + "self_attn.q_proj.weight"], st[p + "self_attn.k_proj.weight"],
                                       st[p + "self_attn.v_proj.weight"]])),
              "o": q(st[p + "self_attn.o_proj.weight"]),
              "ln1": st[p + "input_layernorm.weight"], "ln2": st[p + "post_attention_layernorm.weight"],
              "q_norm": st[p + "self_attn.q_norm.weight"], "k_norm": st[p + "self_attn.k_norm.weight"]}
        if cfg.sparse(i):
            lw["router"] = st[p + "mlp.gate.weight"]
            lw["experts"] = [{"gate_up": q(gu), "down": q(dn)} for gu, dn in expert_weights(st, p, cfg)]
        else:
            lw["gate_up"] = q(np.concatenate([st[p + "mlp.gate_proj.weight"], st[p + "mlp.up_proj.weight"]]))
            lw["down"] = q(st[p + "mlp.down_proj.weight"])
        layers.append(lw)
    final_norm = st["model.norm.weight"]
    state = [StateSpec(n, s, d) for n, (s, d) in
             A.AttnConfig(1, H, HK, D, cfg.layers, opts.max_seq, opts.attention, opts.kv).cache_shapes().items()]

    def head_norm(b, x, heads, weight):
        T = x.shape[0]
        return b.reshape(core.rmsnorm(b, b.reshape(x, [T, heads, D]), weight, cfg.eps), [T, heads * D])

    def body(ctx, tokens, st_vals, pos, last):
        b = ctx.b
        T = tokens.shape[0]
        attn = A.AttnConfig(T, H, HK, D, cfg.layers, opts.max_seq, opts.attention, opts.kv)
        cos_v, sin_v = ctx.weight("rope.cos", cos), ctx.weight("rope.sin", sin)
        if cfg.tied:
            x = E.tied_lookup(ctx, tokens, "lm_head", head, opts.limits)
        else:
            x = E.dense_lookup(ctx, tokens, "embed", embed)
        qd, kvd = H * D, HK * D
        for i, lw in enumerate(layers):
            h = core.rmsnorm(b, x, ctx.weight(f"l{i}.ln1", lw["ln1"]), cfg.eps)
            qkv = L.linear(ctx, h, f"l{i}.qkv", lw["qkv"])
            qq = head_norm(b, b.extract_slice(qkv, [0, 0], [T, qd]), H, ctx.weight(f"l{i}.q_norm", lw["q_norm"]))
            kk = head_norm(b, b.extract_slice(qkv, [0, qd], [T, kvd]), HK, ctx.weight(f"l{i}.k_norm", lw["k_norm"]))
            vv = b.extract_slice(qkv, [0, qd + kvd], [T, kvd])
            qq = E.rope(ctx, qq, H, D, pos, cos_v, sin_v)
            kk = E.rope(ctx, kk, HK, D, pos, cos_v, sin_v)
            o = A.attention(ctx, attn, qq, kk, vv, st_vals, b.index(i), pos)
            x = core.add(b, x, L.linear(ctx, o, f"l{i}.o", lw["o"]))
            h = core.rmsnorm(b, x, ctx.weight(f"l{i}.ln2", lw["ln2"]), cfg.eps)
            if "experts" in lw:
                y = MOE.moe(ctx, h, f"l{i}.moe", lw["router"], lw["experts"], cfg.top_k, cfg.norm_topk, cfg.moe_inter)
            else:
                gu = L.linear(ctx, h, f"l{i}.gate_up", lw["gate_up"])
                y = L.linear(ctx, core.silu_mul(b, b.extract_slice(gu, [0, 0], [T, cfg.inter]),
                                                b.extract_slice(gu, [0, cfg.inter], [T, cfg.inter])),
                             f"l{i}.down", lw["down"])
            x = core.add(b, x, y)
        x = core.rmsnorm(b, x, ctx.weight("norm", final_norm), cfg.eps)
        if last is not None:
            x = b.extract_slice(x, [last, 0], [1, cfg.hidden])
        return L.linear(ctx, x, "lm_head", head)

    build_entry_points(graph, state, cfg.vocab, opts.chunk, body)
    graph.meta.update({"family": "qwen3_moe", "config": {k: v for k, v in vars(cfg).items() if k != "mlp_only"},
                       "options": opts.describe(), "eos": cfg.eos, "source": source})
    if (root / "tokenizer.json").exists():
        graph.files["tokenizer.json"] = root / "tokenizer.json"
    return graph, cfg
