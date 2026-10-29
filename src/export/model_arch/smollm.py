"""SmolLM2 (Llama architecture): tied embedding, RMSNorm, RoPE, GQA, SwiGLU.

Built only from src/export/compilers/mlir/ops. Per layer the NPU runs three tasks
(fused QKV, output, fused gate/up, down = four) and the CPU everything else.
The pinned quantized ONNX export is a separate architecture (smollm_onnx.py).
"""

from __future__ import annotations

from src.export.compilers.mlir.ops import attention as A, core, embedding as E, linear as L
from src.export.ir.graph import ModelGraph, StateSpec, build_prefill_decode_graphs, new_graph
from src.export.ir.options import ShapeOptions
from src.export.ir.quantized import QuantizedModel
from src.export.ir.source import SourceModel
from src.export.model_arch.base import Architecture
from src.export.quantize.plan import WeightSpec

DEFAULT_SOURCE = "HuggingFaceTB/SmolLM2-135M-Instruct"


class Config:
    def __init__(self, hf: dict) -> None:
        self.hidden = hf["hidden_size"]
        self.inter = hf["intermediate_size"]
        self.heads = hf["num_attention_heads"]
        self.kv_heads = hf.get("num_key_value_heads", self.heads)
        self.head_dim = hf.get("head_dim") or self.hidden // self.heads
        self.layers = hf["num_hidden_layers"]
        self.eps = hf["rms_norm_eps"]
        self.theta = hf.get("rope_theta", 10000.0)
        self.vocab = hf["vocab_size"]
        self.tied = hf.get("tie_word_embeddings", True)
        self.eos = hf.get("eos_token_id")


class SmolLM(Architecture):
    name = "smollm"
    title = "SmolLM"
    formats = ("safetensors",)
    default_source = DEFAULT_SOURCE

    def parse_config(self, src: SourceModel) -> Config:
        return Config(src.config)

    def group_dims(self, cfg: Config) -> list[int]:
        return [cfg.hidden, cfg.inter, cfg.heads * cfg.head_dim]

    def awq_spec(self, cfg: Config):
        from src.export.quantize.awq import FoldSpec

        layer = "model.layers.{i}."
        return FoldSpec(layer + "self_attn.q_proj", layer + "self_attn.o_proj", layer + "mlp.gate_proj",
                        layer + "mlp.down_proj", cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim,
                        cfg.inter, cfg.heads, cfg.kv_heads, cfg.head_dim)

    def weight_plan(self, src: SourceModel, cfg: Config) -> list[WeightSpec]:
        specs = [WeightSpec("embed", ("model.embed_tokens.weight",), "embed")]
        for i in range(cfg.layers):
            p = f"model.layers.{i}."
            specs += [
                WeightSpec(f"l{i}.qkv", (p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                                         p + "self_attn.v_proj.weight"), "linear", layer=i, role="qkv"),
                WeightSpec(f"l{i}.o", (p + "self_attn.o_proj.weight",), "linear", layer=i, role="o"),
                WeightSpec(f"l{i}.gate_up", (p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"), "linear",
                           layer=i, role="gate_up"),
                WeightSpec(f"l{i}.down", (p + "mlp.down_proj.weight",), "linear", layer=i, role="down"),
                WeightSpec(f"l{i}.ln1", (p + "input_layernorm.weight",), "float", layer=i, role="ln1"),
                WeightSpec(f"l{i}.ln2", (p + "post_attention_layernorm.weight",), "float", layer=i, role="ln2"),
            ]
        specs.append(WeightSpec("norm", ("model.norm.weight",), "float"))
        return specs

    def to_mlir(self, q: QuantizedModel, shape: ShapeOptions) -> ModelGraph:
        cfg, p = q.config, q.params
        graph = new_graph("smollm")
        H, D = cfg.heads, cfg.head_dim
        emb = p["embed"]
        cos, sin = E.rope_tables(shape.max_seq, D, cfg.theta)

        # Describe the KV cache buffers passed to both entry points.
        kv_shapes = A.AttnConfig(1, H, cfg.kv_heads, D, cfg.layers, shape.max_seq, shape.attention,
                                 shape.kv).cache_shapes()
        state = [StateSpec(n, s, d) for n, (s, d) in kv_shapes.items()]

        # The same forward body handles one decode token or a prefill chunk.
        def body(ctx, tokens, st_vals, pos, last):
            b = ctx.b
            T = tokens.shape[0]
            attn = A.AttnConfig(T, H, cfg.kv_heads, D, cfg.layers, shape.max_seq, shape.attention, shape.kv)
            cos_v = ctx.weight("rope.cos", cos)
            sin_v = ctx.weight("rope.sin", sin)
            x = E.tied_lookup(ctx, tokens, "lm_head", emb, shape.limits)
            q_dim, kv_dim = H * D, cfg.kv_heads * D
            for i in range(cfg.layers):
                h = core.rmsnorm(b, x, ctx.weight(f"l{i}.ln1", p[f"l{i}.ln1"]), cfg.eps)
                qkv = L.linear(ctx, h, f"l{i}.qkv", p[f"l{i}.qkv"])
                qq = b.extract_slice(qkv, [0, 0], [T, q_dim])
                k = b.extract_slice(qkv, [0, q_dim], [T, kv_dim])
                v = b.extract_slice(qkv, [0, q_dim + kv_dim], [T, kv_dim])
                qq = E.rope(ctx, qq, H, D, pos, cos_v, sin_v)
                k = E.rope(ctx, k, cfg.kv_heads, D, pos, cos_v, sin_v)
                layer = b.index(i)
                o = A.attention(ctx, attn, qq, k, v, st_vals, layer, pos)
                x = core.add(b, x, L.linear(ctx, o, f"l{i}.o", p[f"l{i}.o"]))
                h = core.rmsnorm(b, x, ctx.weight(f"l{i}.ln2", p[f"l{i}.ln2"]), cfg.eps)
                gu = L.linear(ctx, h, f"l{i}.gate_up", p[f"l{i}.gate_up"])
                act = core.silu_mul(b, b.extract_slice(gu, [0, 0], [T, cfg.inter]),
                                    b.extract_slice(gu, [0, cfg.inter], [T, cfg.inter]))
                x = core.add(b, x, L.linear(ctx, act, f"l{i}.down", p[f"l{i}.down"]))
            x = core.rmsnorm(b, x, ctx.weight("norm", p["norm"]), cfg.eps)
            if last is not None:
                x = b.extract_slice(x, [last, 0], [1, cfg.hidden])
            return L.linear(ctx, x, "lm_head", emb)

        # Emit separate decode and fixed-chunk prefill functions from that body.
        build_prefill_decode_graphs(graph, state, cfg.vocab, shape.chunk, body)
        # Keep runtime options and the tokenizer alongside the compiled graph.
        graph.meta.update({"family": "llama", "config": vars(cfg), "options": self.describe_options(q, shape),
                           "eos": cfg.eos, "source": q.source.source})
        graph.files["tokenizer.json"] = q.source.root / "tokenizer.json"
        return graph


ARCH = SmolLM()
