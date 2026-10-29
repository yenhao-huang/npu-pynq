"""Qwen3-MoE: Qwen3 attention with sparse mixture-of-experts MLPs.

Router (FP32, CPU), top-k selection and the weighted combine are CPU code;
each expert's gate/up and down projections are NPU tasks guarded by an
scf.if, so only routed experts run. Layers listed in mlp_only_layers (or
off the sparse step) keep a dense SwiGLU MLP.
"""

from __future__ import annotations

from src.export.compilers.mlir.ops import attention as A, core, embedding as E, linear as L, moe as MOE
from src.export.ir.graph import ModelGraph, StateSpec, build_prefill_decode_graphs, new_graph
from src.export.ir.options import ShapeOptions
from src.export.ir.data import QuantizedModel, SourceModel
from src.export.model_arch.base import Architecture, tokenizer_files
from src.export.quantize.plan import WeightSpec

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


def expert_specs(tensors, p: str, cfg: Config, layer: int) -> list[WeightSpec]:
    """Each expert's fused gate/up and down projection, from per-expert
    tensors or from tensors stacked over experts."""
    specs = []
    stacked = p + "mlp.experts.gate_up_proj" in tensors
    for e in range(cfg.experts):
        name = f"l{layer}.experts.{e}."
        if stacked:
            pick = lambda a, e=e: a[e]  # noqa: E731
            specs += [WeightSpec(name + "gate_up", (p + "mlp.experts.gate_up_proj",), "linear", transform=pick,
                                 layer=layer),
                      WeightSpec(name + "down", (p + "mlp.experts.down_proj",), "linear", transform=pick,
                                 layer=layer)]
        else:
            q = f"{p}mlp.experts.{e}."
            specs += [WeightSpec(name + "gate_up", (q + "gate_proj.weight", q + "up_proj.weight"), "linear",
                                 layer=layer),
                      WeightSpec(name + "down", (q + "down_proj.weight",), "linear", layer=layer)]
    return specs


class Qwen3Moe(Architecture):
    name = "qwen3_moe"
    title = "Qwen3-MoE"
    default_source = DEFAULT_SOURCE

    def parse_config(self, src: SourceModel) -> Config:
        return Config(src.config)

    def group_dims(self, cfg: Config) -> list[int]:
        return [cfg.hidden, cfg.moe_inter, cfg.inter, cfg.heads * cfg.head_dim]

    def weight_plan(self, src: SourceModel, cfg: Config) -> list[WeightSpec]:
        specs = [WeightSpec("embed", ("model.embed_tokens.weight",), "embed")]
        if not cfg.tied:
            specs.append(WeightSpec("lm_head", ("lm_head.weight",), "embed"))
        for i in range(cfg.layers):
            p = f"model.layers.{i}."
            specs += [
                WeightSpec(f"l{i}.qkv", (p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                                         p + "self_attn.v_proj.weight"), "linear", layer=i),
                WeightSpec(f"l{i}.o", (p + "self_attn.o_proj.weight",), "linear", layer=i),
                WeightSpec(f"l{i}.ln1", (p + "input_layernorm.weight",), "raw", layer=i),
                WeightSpec(f"l{i}.ln2", (p + "post_attention_layernorm.weight",), "raw", layer=i),
                WeightSpec(f"l{i}.q_norm", (p + "self_attn.q_norm.weight",), "raw", layer=i),
                WeightSpec(f"l{i}.k_norm", (p + "self_attn.k_norm.weight",), "raw", layer=i),
            ]
            if cfg.sparse(i):
                specs.append(WeightSpec(f"l{i}.router", (p + "mlp.gate.weight",), "fp32", layer=i))
                specs += expert_specs(src.tensors, p, cfg, i)
            else:
                specs += [WeightSpec(f"l{i}.gate_up", (p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"),
                                     "linear", layer=i),
                          WeightSpec(f"l{i}.down", (p + "mlp.down_proj.weight",), "linear", layer=i)]
        specs.append(WeightSpec("norm", ("model.norm.weight",), "raw"))
        return specs

    def to_mlir(self, q: QuantizedModel, shape: ShapeOptions) -> ModelGraph:
        cfg, p = q.config, q.params
        graph = new_graph("qwen3_moe")
        H, HK, D = cfg.heads, cfg.kv_heads, cfg.head_dim
        embed = p["embed"]
        head = embed if cfg.tied else p["lm_head"]
        cos, sin = E.rope_tables(shape.max_seq, D, cfg.theta)
        state = [StateSpec(n, s, d) for n, (s, d) in
                 A.AttnConfig(1, H, HK, D, cfg.layers, shape.max_seq, shape.attention, shape.kv).cache_shapes().items()]

        def head_norm(b, x, heads, weight):
            T = x.shape[0]
            return b.reshape(core.rmsnorm(b, b.reshape(x, [T, heads, D]), weight, cfg.eps), [T, heads * D])

        def body(ctx, tokens, st_vals, pos, last):
            b = ctx.b
            T = tokens.shape[0]
            attn = A.AttnConfig(T, H, HK, D, cfg.layers, shape.max_seq, shape.attention, shape.kv)
            cos_v, sin_v = ctx.weight("rope.cos", cos), ctx.weight("rope.sin", sin)
            if cfg.tied:
                x = E.tied_lookup(ctx, tokens, "lm_head", head, shape.limits)
            else:
                x = E.dense_lookup(ctx, tokens, "embed", embed)
            qd, kvd = H * D, HK * D
            for i in range(cfg.layers):
                h = core.rmsnorm(b, x, ctx.weight(f"l{i}.ln1", p[f"l{i}.ln1"]), cfg.eps)
                qkv = L.linear(ctx, h, f"l{i}.qkv", p[f"l{i}.qkv"])
                qq = head_norm(b, b.extract_slice(qkv, [0, 0], [T, qd]), H,
                               ctx.weight(f"l{i}.q_norm", p[f"l{i}.q_norm"]))
                kk = head_norm(b, b.extract_slice(qkv, [0, qd], [T, kvd]), HK,
                               ctx.weight(f"l{i}.k_norm", p[f"l{i}.k_norm"]))
                vv = b.extract_slice(qkv, [0, qd + kvd], [T, kvd])
                qq = E.rope(ctx, qq, H, D, pos, cos_v, sin_v)
                kk = E.rope(ctx, kk, HK, D, pos, cos_v, sin_v)
                o = A.attention(ctx, attn, qq, kk, vv, st_vals, b.index(i), pos)
                x = core.add(b, x, L.linear(ctx, o, f"l{i}.o", p[f"l{i}.o"]))
                h = core.rmsnorm(b, x, ctx.weight(f"l{i}.ln2", p[f"l{i}.ln2"]), cfg.eps)
                if cfg.sparse(i):
                    experts = [{"gate_up": p[f"l{i}.experts.{e}.gate_up"], "down": p[f"l{i}.experts.{e}.down"]}
                               for e in range(cfg.experts)]
                    y = MOE.moe(ctx, h, f"l{i}.moe", p[f"l{i}.router"], experts, cfg.top_k, cfg.norm_topk,
                                cfg.moe_inter)
                else:
                    gu = L.linear(ctx, h, f"l{i}.gate_up", p[f"l{i}.gate_up"])
                    y = L.linear(ctx, core.silu_mul(b, b.extract_slice(gu, [0, 0], [T, cfg.inter]),
                                                    b.extract_slice(gu, [0, cfg.inter], [T, cfg.inter])),
                                 f"l{i}.down", p[f"l{i}.down"])
                x = core.add(b, x, y)
            x = core.rmsnorm(b, x, ctx.weight("norm", p["norm"]), cfg.eps)
            if last is not None:
                x = b.extract_slice(x, [last, 0], [1, cfg.hidden])
            return L.linear(ctx, x, "lm_head", head)

        build_prefill_decode_graphs(graph, state, cfg.vocab, shape.chunk, body)
        graph.meta.update({"family": "qwen3_moe", "config": {k: v for k, v in vars(cfg).items() if k != "mlp_only"},
                           "options": self.describe_options(q, shape), "eos": cfg.eos, "source": q.source.source})
        graph.files.update(tokenizer_files(q.source.root, "tokenizer.json", optional=True))
        return graph


ARCH = Qwen3Moe()
