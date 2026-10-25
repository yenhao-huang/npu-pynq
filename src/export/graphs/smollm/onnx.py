"""SmolLM2 from the pinned quantized ONNX export (onnx-source.json)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.export.compilers.mlir.builder import identity_map, memref_type
from src.export.compilers.mlir.ops import linear as L
from src.export.compilers.mlir.ops.context import Ctx
from src.export.core.graph import new_graph
from src.export.core.model_loader import load_weights

ONNX_PIN = Path(__file__).with_name('onnx-source.json')


def build_quantized_onnx(source, prefill_lengths=(36, 40, 42), max_seq=512, layers=30):
    """Build SmolLM's graphs from the pinned INT8 ONNX export, keeping its
    quantization unchanged; float operators run through ONNX Runtime."""
    source = Path(source)
    onnx_path = source if source.is_file() else source / 'model_int8.onnx'
    source = onnx_path.parent
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    pinned = json.loads(ONNX_PIN.read_text())
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
