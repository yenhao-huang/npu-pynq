"""SmolLM2 (Llama architecture): tied embedding, RMSNorm, RoPE, GQA, SwiGLU.

Built only from src/compiler/ops. Per layer the NPU runs three tasks
(fused QKV, output, fused gate/up, down = four) and the CPU everything else.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from src.compiler.ops import attention as A, core, embedding as E, linear as L
from src.inference.graph import LLMOptions, StateSpec, build_prefill_decode_graphs, choose_group, new_graph
from src.inference.weights import SafeTensors, resolve_hf, load_weights, onnx_asset_dir
import hashlib
from src.compiler.mlir_builder import identity_map, memref_type
from src.compiler.ops.context import Ctx


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
    if awq is not None:
        mats, norms = awq(layer, mats, norms)
    q = {k: L.quantize_weight(v, opts.bits, opts.group) for k, v in mats.items()}
    q.update({k: v.astype(np.float32) for k, v in norms.items()})
    return q


def awq_spec(cfg: Config):
    from src.inference.quant.awq import FoldSpec

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
        graph, source_model = _build_quantized_onnx(path if path.suffix == '.onnx' else source_root,
                                      prefill_lengths, opts.max_seq, opts.layers or 30)
        from src.compiler.export import attach_float_runtime
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
        from src.inference.quant import awq as AWQ
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


def _build_quantized_onnx(source, prefill_lengths=(36, 40, 42), max_seq=512, layers=30):
    source = Path(source)
    onnx_path = source if source.is_file() else source / 'model_int8.onnx'
    source = onnx_path.parent
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    pinned = json.loads((Path(__file__).resolve().parents[2] /
                        'compiler/onnx-source.json').read_text())
    if digest != pinned['files']['model_int8.onnx']['sha256']:
        raise ValueError('This frontend requires the pinned pinned ONNX artifact')
    weights = load_weights(onnx_path)
    model = weights.model
    integer = [n for n in model.graph.node if n.op_type == 'MatMulInteger']
    if len(integer) != 210 or not 1 <= layers <= 30:
        raise ValueError('Unexpected ONNX topology or requested layer count')
    for node in integer:
        if np.any(weights[node.input[3]] != 0):
            raise ValueError('Projection weight zero points must be zero: ' + node.name)
    # Fusion is permitted only for contractions sharing exact quantized inputs.
    for layer in range(layers):
        for names in (('attn/q_proj', 'attn/k_proj', 'attn/v_proj'),
                      ('mlp/gate_proj', 'mlp/up_proj')):
            nodes = [next(n for n in integer if n.name ==
                f'/model/layers.{layer}/{name}/MatMul_quant') for name in names]
            if len({(n.input[0], n.input[2]) for n in nodes}) != 1:
                raise ValueError('Fusion would change activation quantization')
    lengths = sorted(set(int(t) for t in prefill_lengths))
    if not lengths or any(t < 1 or t > max_seq for t in lengths):
        raise ValueError('Prefill lengths must lie within the context limit')
    graph = new_graph('smollm-onnx')
    hidden, vocab, inter = 576, 49152, 1536
    cache_shape = [layers, 3, max_seq, 64]
    embed = weights['model.embed_tokens.weight_quantized']
    head = ((embed.astype(np.float32) - weights['model.embed_tokens.weight_zero_point']) *
            weights['model.embed_tokens.weight_scale']).T.copy()
    projections = []
    for i in range(layers):
        group = {}
        for label, branches in {'qkv':('attn.q_proj','attn.k_proj','attn.v_proj'),
                               'o':('attn.o_proj',),
                               'gu':('mlp.gate_proj','mlp.up_proj'),
                               'down':('mlp.down_proj',)}.items():
            keys = [f'model.layers.{i}.{branch}.MatMul' for branch in branches]
            matrices = [weights[key + '.weight_quantized'] for key in keys]
            scales = [np.full(matrix.shape[1], weights[key + '.weight_scale'], np.float32)
                      for key, matrix in zip(keys, matrices)]
            group[label] = (np.concatenate(matrices, axis=1), np.concatenate(scales))
        projections.append(group)
    for name, t in [('decode', 1)] + [(f'prefill_{t}', t) for t in lengths]:
        args = [('ws','memref<?xi8>'), ('key',memref_type(cache_shape,'f32')),
                ('value',memref_type(cache_shape,'f32'))]
        args += [('tokens',memref_type([t],'i32')), ('pos','index'),
                 ('logits',memref_type([1,vocab],'f32'))]
        b = graph.module.func(name, args)
        ctx = Ctx(b, graph, b.arg('ws'))
        token_tensor = b.to_tensor(b.arg('tokens'))
        head_value = ctx.weight('onnx.head', head)
        output = b.empty([t,hidden], 'f32')

        def lookup(r):
            row = r.index(0)
            column = r.index(1)
            token = r.extract(token_tensor, [row])
            token = r.index_cast(token)
            return [r.extract(head_value, [column, token])]

        x = b.generic([], [output], [identity_map(2)], ['parallel','parallel'], lookup)[0]
        gamma = ctx.weight('l0.ln1', weights['model.layers.0.input_layernorm.weight'])
        h = L.native(ctx, 'onnx_norm', [x,gamma], [([t,hidden],'f32')])[0]
        cos = ctx.weight('onnx.cos', weights['cos_cache'])
        sin = ctx.weight('onnx.sin', weights['sin_cache'])
        for i, group in enumerate(projections):
            qkv = L.projection(ctx,L.quantize(ctx,h),f'l{i}.qkv',*group['qkv'])
            q = b.extract_slice(qkv,[0,0],[t,576])
            k = b.extract_slice(qkv,[0,576],[t,192])
            v = b.extract_slice(qkv,[0,768],[t,192])
            q = L.native(ctx,'onnx_rope',[q,cos,sin,b.arg('pos')],[([t,576],'f32')])[0]
            k = L.native(ctx,'onnx_rope',[k,cos,sin,b.arg('pos')],[([t,192],'f32')])[0]
            attention = L.native(ctx,'onnx_attention',
                [q,k,v,b.arg('key'),b.arg('value'),b.index(i),b.arg('pos')],
                [([t,hidden],'f32')])[0]
            o = L.projection(ctx,L.quantize(ctx,attention),f'l{i}.o',*group['o'])
            gamma = ctx.weight(f'l{i}.ln2',weights[f'model.layers.{i}.post_attention_layernorm.weight'])
            h, residual = L.native(ctx,'onnx_skip',[x,o,gamma],
                                   [([t,hidden],'f32'),([t,hidden],'f32')])
            gu = L.projection(ctx,L.quantize(ctx,h),f'l{i}.gu',*group['gu'])
            gate = b.extract_slice(gu,[0,0],[t,inter])
            up = b.extract_slice(gu,[0,inter],[t,inter])
            sigmoid = L.native(ctx,'onnx_sigmoid',[gate],[([t,inter],'f32')])[0]
            act = L.multiply(b,L.multiply(b,gate,sigmoid),up)
            down = L.projection(ctx,L.quantize(ctx,act),f'l{i}.down',*group['down'])
            key = (f'model.layers.{i+1}.input_layernorm.weight' if i+1 < 30
                   else 'model.layers.30.final_norm_layernorm.weight')
            gamma = ctx.weight(f'l{i}.next_norm',weights[key])
            h,x = L.native(ctx,'onnx_skip',[residual,down,gamma],
                           [([t,hidden],'f32'),([t,hidden],'f32')])
        last = b.extract_slice(h,[t-1,0],[1,hidden])
        logits = L.native(ctx,'onnx_head',[last,head_value],[([1,vocab],'f32')])[0]
        b.store(logits,b.arg('logits'))
        b.emit('return')
        graph.entry_points[name] = ['weights','key','value','tokens','pos','logits']
    graph.meta.update({'family':'llama','source_format':'onnx','source_sha256':digest,
        'state':[{'name':kind,'shape':cache_shape,'dtype':'f32'} for kind in ('key','value')],
        'vocab':vocab,'prefill_lengths':lengths,'layers':layers,
        'integer_projections_per_forward':4*layers,'original_integer_projections':7*layers,
        'options':{'max_seq':max_seq},'native_float_runtime':'onnxruntime-1.20.1',
        'fusions':['qkv','gate_up'],'onnx_quantization_unchanged':True})
    graph.files['tokenizer.json'] = source/'tokenizer.json'
    graph.files['tokenizer_config.json'] = source/'tokenizer_config.json'
    return graph, model
