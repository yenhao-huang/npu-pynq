"""SmolLM2 (Llama architecture): tied embedding, RMSNorm, RoPE, GQA, SwiGLU.

Built only from src/export/compilers/mlir/ops. Per layer the NPU runs three tasks
(fused QKV, output, fused gate/up, down = four) and the CPU everything else.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from src.export.compilers.mlir.ops import attention as A, core, embedding as E, linear as L
from src.export.core.graph import LLMOptions, StateSpec, build_prefill_decode_graphs, choose_group, new_graph
from src.export.core.model_loader import SafeTensors, resolve_hf, load_weights
from src.export.core.weight_handler import quantize_layer
from src.export.graphs.smollm.onnx import build_quantized_onnx
from src.runtime.core.package import onnx_asset_dir


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


def quantize_layer_weights(st: SafeTensors, cfg: Config, layer: int, opts: LLMOptions, awq=None) -> dict:
    p = f"model.layers.{layer}."
    w = lambda n: st[p + n]  # noqa: E731
    qkv = np.concatenate([w("self_attn.q_proj.weight"), w("self_attn.k_proj.weight"),
                          w("self_attn.v_proj.weight")], axis=0)
    gate_up = np.concatenate([w("mlp.gate_proj.weight"), w("mlp.up_proj.weight")], axis=0)
    mats = {"qkv": qkv, "o": w("self_attn.o_proj.weight"), "gate_up": gate_up, "down": w("mlp.down_proj.weight")}
    norms = {"ln1": w("input_layernorm.weight"), "ln2": w("post_attention_layernorm.weight")}
    return quantize_layer(layer, mats, norms, opts.bits, opts.group, awq)


def awq_spec(cfg: Config):
    from src.export.quant.awq import FoldSpec

    layer = "model.layers.{i}."
    return FoldSpec(layer + "self_attn.q_proj", layer + "self_attn.o_proj", layer + "mlp.gate_proj",
                    layer + "mlp.down_proj", cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim,
                    cfg.inter, cfg.heads, cfg.kv_heads, cfg.head_dim)


def build(source: str = DEFAULT_SOURCE, opts: LLMOptions | None = None, awq=None,
          source_format='auto', prefill_lengths=(36, 40, 42)):
    """Prepare SmolLM weights and build its prefill and decode graphs."""
    opts = opts or LLMOptions()
    source = str(source)
    path = Path(source).expanduser()
    if source_format == 'auto':
        source_format = ('onnx' if path.suffix == '.onnx' or
                         (path / 'model_int8.onnx').is_file() or
                         (onnx_asset_dir(path, 'onnx') / 'model_int8.onnx').is_file() else 'safetensors')
    if source_format == 'onnx':
        if awq is not None or opts.awq or opts.bits != 8 or opts.kv != 'f32' or opts.max_seq != 512:
            raise ValueError('The quantized ONNX input retains INT8 weights, FP32 KV and its 512-token contract')
        prepared = path.parent if path.suffix == '.onnx' else path
        nested = onnx_asset_dir(prepared, 'onnx')
        source_root = nested if nested.is_dir() else prepared
        if source_root.name in ('onnx', 'official-onnx'):
            prepared = source_root.parent
        graph, source_model = build_quantized_onnx(path if path.suffix == '.onnx' else source_root,
                                      prefill_lengths, opts.max_seq, opts.layers or 30)
        from src.export.core.onnx_source import attach_float_runtime
        attach_float_runtime(graph, prepared, source_model)
        return graph, Config(json.loads((source_root / 'config.json').read_text(encoding='utf-8')))
    if source_format != 'safetensors':
        raise ValueError('Unsupported SmolLM source format: ' + source_format)
    # Read the model configuration and weights from a local path or HF cache.
    root = resolve_hf(source)
    weight_source = root
    if root.is_file():
        root = root.parent
    cfg = Config(json.loads((root / "config.json").read_text()))
    st = load_weights(weight_source)
    # Start an empty graph, then prepare values shared by all layers.
    graph = new_graph("smollm")
    H, D = cfg.heads, cfg.head_dim
    emb = L.quantize_weight(st["model.embed_tokens.weight"], opts.embed_bits, opts.group)
    cos, sin = E.rope_tables(opts.max_seq, D, cfg.theta)
    if opts.layers:
        cfg.layers = opts.layers
    opts.group = choose_group(opts.group, [cfg.hidden, cfg.inter, cfg.heads * cfg.head_dim])
    # Calibrate AWQ when requested, then quantize each decoder layer's weights.
    if opts.awq and awq is None:
        from src.export.quant import awq as AWQ
        awq = AWQ.hook_for(root, AWQ.calibration_ids(root), awq_spec(cfg), cfg.layers, opts.bits, opts.group)
    layers = [quantize_layer_weights(st, cfg, i, opts, awq) for i in range(cfg.layers)]
    final_norm = st["model.norm.weight"].astype(np.float32)

    # Describe the KV cache buffers passed to both entry points.
    kv_shapes = A.AttnConfig(1, H, cfg.kv_heads, D, cfg.layers, opts.max_seq, opts.attention, opts.kv).cache_shapes()
    state = [StateSpec(n, s, d) for n, (s, d) in kv_shapes.items()]

    # The same forward body handles one decode token or a prefill chunk.
    def body(ctx, tokens, st_vals, pos, last):
        b = ctx.b
        T = tokens.shape[0]
        attn = A.AttnConfig(T, H, cfg.kv_heads, D, cfg.layers, opts.max_seq, opts.attention, opts.kv)
        cos_v = ctx.weight("rope.cos", cos)
        sin_v = ctx.weight("rope.sin", sin)
        x = E.tied_lookup(ctx, tokens, "lm_head", emb, opts.limits)
        q_dim, kv_dim = H * D, cfg.kv_heads * D
        for i, lw in enumerate(layers):
            h = core.rmsnorm(b, x, ctx.weight(f"l{i}.ln1", lw["ln1"]), cfg.eps)
            qkv = L.linear(ctx, h, f"l{i}.qkv", lw["qkv"])
            q = b.extract_slice(qkv, [0, 0], [T, q_dim])
            k = b.extract_slice(qkv, [0, q_dim], [T, kv_dim])
            v = b.extract_slice(qkv, [0, q_dim + kv_dim], [T, kv_dim])
            q = E.rope(ctx, q, H, D, pos, cos_v, sin_v)
            k = E.rope(ctx, k, cfg.kv_heads, D, pos, cos_v, sin_v)
            layer = b.index(i)
            o = A.attention(ctx, attn, q, k, v, st_vals, layer, pos)
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

    # Emit separate decode and fixed-chunk prefill functions from that body.
    build_prefill_decode_graphs(graph, state, cfg.vocab, opts.chunk, body)
    # Keep runtime options and the tokenizer alongside the compiled graph.
    graph.meta.update({"family": "llama", "config": vars(cfg), "options": opts.describe(),
                       "eos": cfg.eos, "source": source})
    graph.files["tokenizer.json"] = root / "tokenizer.json"
    return graph, cfg
