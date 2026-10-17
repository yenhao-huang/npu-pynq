"""Qwen3.5 (text): hybrid linear attention.

Three of every four layers are Gated DeltaNet linear attention (causal conv
+ gated delta rule, recurrent state instead of a KV cache); the fourth is
full attention with an output gate, per-head zero-centred q/k RMSNorm and
partial RoPE (25% of the head dimension). RMSNorms are zero-centred
(x_hat * (1 + w)). All projections are NPU tasks; per layer the linear
attention fuses its q/k/v, z, b and a projections into one.
"""

from __future__ import annotations

import json

import numpy as np

from src.compiler.mlir_builder import Region, Value, identity_map, map_of
from src.compiler.ops import attention as A, core, embedding as E, linear as L, linear_attention as G
from src.inference.graph import LLMOptions, StateSpec, build_prefill_decode_graphs, choose_group, new_graph
from src.inference.weights import SafeTensors, resolve_hf

DEFAULT_SOURCE = "Qwen/Qwen3.5-0.8B"


class Config:
    def __init__(self, hf: dict) -> None:
        t = hf.get("text_config", hf)
        self.hidden = t["hidden_size"]
        self.inter = t["intermediate_size"]
        self.heads = t["num_attention_heads"]
        self.kv_heads = t["num_key_value_heads"]
        self.head_dim = t["head_dim"]
        self.layers = t["num_hidden_layers"]
        self.layer_types = list(t["layer_types"])
        self.eps = t["rms_norm_eps"]
        rope = t.get("rope_parameters") or {}
        self.theta = rope.get("rope_theta", t.get("rope_theta", 10000000.0))
        self.rotary_dim = int(self.head_dim * rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0)))
        self.k_heads = t["linear_num_key_heads"]
        self.v_heads = t["linear_num_value_heads"]
        self.dk = t["linear_key_head_dim"]
        self.dv = t["linear_value_head_dim"]
        self.conv_kernel = t["linear_conv_kernel_dim"]
        self.vocab = t["vocab_size"]
        self.eos = t.get("eos_token_id", hf.get("eos_token_id"))


def build(source: str = DEFAULT_SOURCE, opts: LLMOptions | None = None, awq=None):
    opts = opts or LLMOptions()
    root = resolve_hf(source)
    cfg = Config(json.loads((root / "config.json").read_text()))
    if opts.layers:
        cfg.layers = opts.layers
        cfg.layer_types = cfg.layer_types[: opts.layers]
    opts.group = choose_group(opts.group, [cfg.hidden, cfg.inter, cfg.heads * cfg.head_dim, cfg.v_heads * cfg.dv])
    st = SafeTensors(root)
    pre = "model.language_model."
    w = lambda n: st[pre + n]  # noqa: E731
    q = lambda m: L.quantize_weight(m, opts.bits, opts.group)  # noqa: E731
    H, HK, D = cfg.heads, cfg.kv_heads, cfg.head_dim
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    lin = [i for i, t in enumerate(cfg.layer_types) if t != "full_attention"]
    emb = L.quantize_weight(w("embed_tokens.weight"), opts.embed_bits, opts.group)
    cos, sin = E.rope_tables(opts.max_seq, cfg.rotary_dim, cfg.theta)
    layers = []
    for i in range(cfg.layers):
        p = f"layers.{i}."
        lw = {"type": cfg.layer_types[i], "ln1": w(p + "input_layernorm.weight"),
              "ln2": w(p + "post_attention_layernorm.weight"),
              "gate_up": q(np.concatenate([w(p + "mlp.gate_proj.weight"), w(p + "mlp.up_proj.weight")])),
              "down": q(w(p + "mlp.down_proj.weight"))}
        if lw["type"] == "full_attention":
            qp = w(p + "self_attn.q_proj.weight").reshape(H, 2, D, cfg.hidden)  # per head: query then gate
            lw["proj"] = q(np.concatenate([qp[:, 0].reshape(H * D, -1), qp[:, 1].reshape(H * D, -1),
                                           w(p + "self_attn.k_proj.weight"), w(p + "self_attn.v_proj.weight")]))
            lw["o"] = q(w(p + "self_attn.o_proj.weight"))
            lw["q_norm"] = w(p + "self_attn.q_norm.weight")
            lw["k_norm"] = w(p + "self_attn.k_norm.weight")
        else:
            la = p + "linear_attn."
            lw["proj"] = q(np.concatenate([w(la + "in_proj_qkv.weight"), w(la + "in_proj_z.weight"),
                                           w(la + "in_proj_b.weight"), w(la + "in_proj_a.weight")]))
            lw["out"] = q(w(la + "out_proj.weight"))
            lw["conv"] = w(la + "conv1d.weight").reshape(-1, cfg.conv_kernel)
            lw["neg_A"] = -np.exp(w(la + "A_log").astype(np.float64)).astype(np.float32)
            lw["dt_bias"] = w(la + "dt_bias")
            lw["norm"] = w(la + "norm.weight")
        layers.append(lw)
    final_norm = w("norm.weight")
    delta = lambda T: G.DeltaConfig(T, cfg.k_heads, cfg.v_heads, cfg.dk, cfg.dv, cfg.conv_kernel, len(lin))  # noqa: E731
    attn = lambda T: A.AttnConfig(T, H, HK, D, len(full), opts.max_seq, opts.attention, opts.kv)  # noqa: E731
    state = [StateSpec(n, s, d) for n, (s, d) in attn(1).cache_shapes().items()]
    state += [StateSpec(n, s, d) for n, (s, d) in delta(1).state_shapes().items()]

    def norm(b, x, weight):  # zero-centred RMSNorm
        return core.rmsnorm(b, x, weight, cfg.eps, add_unit_offset=True)

    def head_norm(ctx, x, heads, weight):
        b = ctx.b
        T = x.shape[0]
        return b.reshape(norm(b, b.reshape(x, [T, heads, D]), weight), [T, heads * D])

    def body(ctx, tokens, sv, pos, last):
        b = ctx.b
        T = tokens.shape[0]
        cos_v, sin_v = ctx.weight("rope.cos", cos), ctx.weight("rope.sin", sin)
        x = E.tied_lookup(ctx, tokens, "lm_head", emb, opts.limits)
        for i, lw in enumerate(layers):
            h = norm(b, x, ctx.weight(f"l{i}.ln1", lw["ln1"]))
            proj = L.linear(ctx, h, f"l{i}.proj", lw["proj"])
            if lw["type"] == "full_attention":
                qd, kvd = H * D, HK * D
                qq = head_norm(ctx, b.extract_slice(proj, [0, 0], [T, qd]), H, ctx.weight(f"l{i}.q_norm", lw["q_norm"]))
                gate = b.extract_slice(proj, [0, qd], [T, qd])
                kk = head_norm(ctx, b.extract_slice(proj, [0, 2 * qd], [T, kvd]), HK,
                               ctx.weight(f"l{i}.k_norm", lw["k_norm"]))
                vv = b.extract_slice(proj, [0, 2 * qd + kvd], [T, kvd])
                qq = E.rope(ctx, qq, H, D, pos, cos_v, sin_v, cfg.rotary_dim)
                kk = E.rope(ctx, kk, HK, D, pos, cos_v, sin_v, cfg.rotary_dim)
                o = A.attention(ctx, attn(T), qq, kk, vv, {n: sv[n] for n in ("k", "v", "ks", "vs") if n in sv},
                                b.index(full.index(i)), pos)
                o = b.elementwise([o, core.sigmoid(b, gate)], "f32", lambda r: r.mulf(r.args[0], r.args[1]))
                y = L.linear(ctx, o, f"l{i}.o", lw["o"])
            else:
                dc = delta(T)
                C, vd, HV = dc.channels, dc.value_dim, cfg.v_heads
                mixed = b.extract_slice(proj, [0, 0], [T, C])
                z = b.extract_slice(proj, [0, C], [T, vd])
                bb = b.extract_slice(proj, [0, C + vd], [T, HV])
                aa = b.extract_slice(proj, [0, C + vd + HV], [T, HV])
                beta = core.sigmoid(b, bb)
                neg_a, dt = ctx.weight(f"l{i}.neg_A", lw["neg_A"]), ctx.weight(f"l{i}.dt_bias", lw["dt_bias"])

                def decay(r: Region) -> Value:
                    s = r.addf(r.args[0], r.args[2])
                    sp = r.log(r.addf(r.const(1.0, "f32"), r.exp(s)))
                    sp = r.select(r.cmpf("ogt", s, r.const(20.0, "f32")), s, sp)  # torch softplus threshold
                    return r.mulf(r.args[1], sp)

                g = b.elementwise([aa, neg_a, dt], "f32", decay,
                                  maps=[identity_map(2), map_of(2, ["d1"]), map_of(2, ["d1"])])
                core_out = G.gated_delta(ctx, dc, mixed, ctx.weight(f"l{i}.conv", lw["conv"]), g, beta,
                                         {"conv": sv["conv"], "rec": sv["rec"]}, b.index(lin.index(i)))
                normed = core.rmsnorm(b, core_out, ctx.weight(f"l{i}.gnorm", lw["norm"]), cfg.eps)
                gated = b.elementwise([normed, b.reshape(z, [T, HV, cfg.dv])], "f32",
                                      lambda r: r.mulf(r.args[0], r.divf(r.args[1], r.addf(
                                          r.const(1.0, "f32"), r.exp(r.neg(r.args[1]))))))
                y = L.linear(ctx, b.reshape(gated, [T, vd]), f"l{i}.out", lw["out"])
            x = core.add(b, x, y)
            h = norm(b, x, ctx.weight(f"l{i}.ln2", lw["ln2"]))
            gu = L.linear(ctx, h, f"l{i}.gate_up", lw["gate_up"])
            act = core.silu_mul(b, b.extract_slice(gu, [0, 0], [T, cfg.inter]),
                                b.extract_slice(gu, [0, cfg.inter], [T, cfg.inter]))
            x = core.add(b, x, L.linear(ctx, act, f"l{i}.down", lw["down"]))
        x = norm(b, x, ctx.weight("norm", final_norm))
        if last is not None:
            x = b.extract_slice(x, [last, 0], [1, cfg.hidden])
        return L.linear(ctx, x, "lm_head", emb)

    build_prefill_decode_graphs(graph := new_graph("qwen3_5"), state, cfg.vocab, opts.chunk, body)
    graph.meta.update({"family": "qwen3_5", "config": {k: v for k, v in vars(cfg).items() if k != "layer_types"},
                       "layer_types": cfg.layer_types, "options": opts.describe(), "eos": cfg.eos, "source": source})
    graph.files["tokenizer.json"] = root / "tokenizer.json"
    return graph, cfg
