"""SmolLM2 from the pinned quantized ONNX export (onnx-source.json).

The ONNX quantization is kept unchanged: load verifies the pin, imports the
graph through ONNX-MLIR and checks the topology; quantize only regroups the
INT8 projections that share an activation quantization (q/k/v and gate/up);
to_mlir compiles the CPU operators from the imported ONNX semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.export.compilers.mlir.builder import identity_map, memref_type
from src.export.compilers.mlir.ops.attention import AttnConfig
from src.export.compilers.mlir.ops.context import Ctx
from src.export.compilers.onnx_mlir import lowering
from src.export.ir.graph import ModelGraph, new_graph
from src.export.ir.options import QuantRecipe, ShapeOptions
from src.export.ir.data import QuantizedModel, SourceModel
from src.export.load import load_weights
from src.export.load.onnx import import_graph, imported_node, verify_pin
from src.export.model_arch.base import Architecture
from src.export.model_arch.smollm.arch import Config
from src.export.quantize.plan import AwqHook
from src.export.quantize.preserve import dequantize, fuse_columns
from src.runtime.core.package import onnx_asset_dir

ONNX_PIN = Path(__file__).with_name('onnx-source.json')
LAYERS = 30
HIDDEN, VOCAB, INTER = 576, 49152, 1536
# Fused parameter -> the original ONNX projections it concatenates.
FUSIONS = {'qkv': ('attn.q_proj', 'attn.k_proj', 'attn.v_proj'), 'o': ('attn.o_proj',),
           'gu': ('mlp.gate_proj', 'mlp.up_proj'), 'down': ('mlp.down_proj',)}


class SmolLMOnnx(Architecture):
    name = 'smollm_onnx'
    title = 'SmolLM'
    formats = ('onnx',)

    def validate(self, recipe: QuantRecipe, shape: ShapeOptions, awq: AwqHook | None) -> None:
        if awq is not None or recipe.awq or recipe.bits != 8 or shape.kv != 'f32' or shape.max_seq != 512:
            raise ValueError('The quantized ONNX input retains INT8 weights, FP32 KV and its 512-token contract')

    # load

    def load(self, source: str, fmt: str, layers: int | None, onnx_import_dir=None, **kw) -> SourceModel:
        path = Path(source).expanduser()
        prepared = path.parent if path.suffix == '.onnx' else path
        nested = onnx_asset_dir(prepared, 'onnx')
        source_root = nested if nested.is_dir() else prepared
        given = path if path.suffix == '.onnx' else source_root
        onnx_path = given if given.is_file() else given / 'model_int8.onnx'
        layers = layers or LAYERS
        digest = verify_pin(onnx_path, ONNX_PIN)
        _, report, import_dir, nodes = import_graph(onnx_path, onnx_import_dir)
        weights = load_weights(onnx_path)
        integer = [n for n in weights.model.graph.node if n.op_type == 'MatMulInteger']
        if len(integer) != 7 * LAYERS or not 1 <= layers <= LAYERS:
            raise ValueError('Unexpected ONNX topology or requested layer count')
        for node in integer:
            if np.any(weights[node.input[3]] != 0):
                raise ValueError('Projection weight zero points must be zero: ' + node.name)
        # Fusion is permitted only for contractions sharing exact quantized inputs.
        for layer in range(layers):
            for names in (('attn/q_proj', 'attn/k_proj', 'attn/v_proj'), ('mlp/gate_proj', 'mlp/up_proj')):
                contractions = [imported_node(nodes, f'/model/layers.{layer}/{name}/MatMul_quant',
                                              'onnx.MatMulInteger') for name in names]
                if len({(n['inputs'][0], n['inputs'][2]) for n in contractions}) != 1:
                    raise ValueError('Fusion would change activation quantization')
        config = json.loads((source_root / 'config.json').read_text(encoding='utf-8'))
        return SourceModel(self.name, fmt, source, onnx_path.parent, config, weights, layers,
                           {'sha256': digest, 'import_report': report, 'import_dir': import_dir, 'nodes': nodes})

    # quantize: preserve the source's INT8 weights

    def parse_config(self, src: SourceModel) -> Config:
        return Config(src.config)

    def config(self, src: SourceModel) -> Config:
        return self.parse_config(src)   # the layer limit selects ONNX layers; config.json stays as exported

    def quantize(self, src: SourceModel, recipe: QuantRecipe, awq: AwqHook | None = None) -> QuantizedModel:
        w = src.tensors
        params = {'onnx.head': dequantize(w['model.embed_tokens.weight_quantized'],
                                          w['model.embed_tokens.weight_zero_point'],
                                          w['model.embed_tokens.weight_scale']).T.copy(),
                  'l0.ln1': w['model.layers.0.input_layernorm.weight'],
                  'onnx.cos': w['cos_cache'], 'onnx.sin': w['sin_cache']}
        for i in range(src.layers):
            for label, branches in FUSIONS.items():
                params[f'l{i}.{label}'] = fuse_columns(w, [f'model.layers.{i}.{b}.MatMul' for b in branches])
            params[f'l{i}.ln2'] = w[f'model.layers.{i}.post_attention_layernorm.weight']
            params[f'l{i}.next_norm'] = w[f'model.layers.{i + 1}.input_layernorm.weight' if i + 1 < LAYERS
                                          else f'model.layers.{LAYERS}.final_norm_layernorm.weight']
        return QuantizedModel(src, recipe, self.config(src), params, {'fusions': ['qkv', 'gate_up']})

    # to_mlir

    def to_mlir(self, q: QuantizedModel, shape: ShapeOptions) -> ModelGraph:
        src, p = q.source, q.params
        nodes, layers, max_seq = src.provenance['nodes'], src.layers, shape.max_seq
        lengths = sorted(set(int(t) for t in shape.prefill_lengths))
        if not lengths or any(t < 1 or t > max_seq for t in lengths):
            raise ValueError('Prefill lengths must lie within the context limit')
        graph = new_graph('smollm-onnx')
        cache_shape = [layers, 3, max_seq, 64]
        for name, t in [('decode', 1)] + [(f'prefill_{t}', t) for t in lengths]:
            args = [('ws', 'memref<?xi8>'), ('key', memref_type(cache_shape, 'f32')),
                    ('value', memref_type(cache_shape, 'f32'))]
            args += [('tokens', memref_type([t], 'i32')), ('pos', 'index'),
                     ('logits', memref_type([1, VOCAB], 'f32'))]
            b = graph.module.func(name, args)
            ctx = Ctx(b, graph, b.arg('ws'))

            def custom(name, inputs, config=None):
                return lowering.custom(ctx, imported_node(nodes, name, 'onnx.Custom'), inputs, config=config)

            def quantize(name, value):
                imported_node(nodes, name, 'onnx.DynamicQuantizeLinear')
                return lowering.dynamic_quantize(b, value)

            def projection(value, name):
                return lowering.projection(ctx, value, name, p[name].q, p[name].scale)

            token_tensor = b.to_tensor(b.arg('tokens'))
            head_value = ctx.weight('onnx.head', p['onnx.head'])
            output = b.empty([t, HIDDEN], 'f32')

            def lookup(r):
                row = r.index(0)
                column = r.index(1)
                token = r.extract(token_tensor, [row])
                token = r.index_cast(token)
                return [r.extract(head_value, [column, token])]

            x = b.generic([], [output], [identity_map(2)], ['parallel', 'parallel'], lookup)[0]
            gamma = ctx.weight('l0.ln1', p['l0.ln1'])
            h = custom('/model/layers.0/input_layernorm/LayerNorm', [x, gamma])[0]
            cos = ctx.weight('onnx.cos', p['onnx.cos'])
            sin = ctx.weight('onnx.sin', p['onnx.sin'])
            for i in range(layers):
                prefix = f'/model/layers.{i}'
                qkv = projection(quantize(prefix + '/input_layernorm/output_0_QuantizeLinear', h), f'l{i}.qkv')
                qq = b.extract_slice(qkv, [0, 0], [t, 576])
                k = b.extract_slice(qkv, [0, 576], [t, 192])
                v = b.extract_slice(qkv, [0, 768], [t, 192])
                qq = custom(prefix + '/attn/q_rotary/RotaryEmbedding', [qq, cos, sin, b.arg('pos')])[0]
                k = custom(prefix + '/attn/k_rotary/RotaryEmbedding', [k, cos, sin, b.arg('pos')])[0]
                attention = custom(prefix + '/attn/GroupQueryAttention',
                    [qq, k, v, {'k': b.arg('key'), 'v': b.arg('value')}, b.index(i), b.arg('pos')],
                    AttnConfig(t, 9, 3, 64, layers, max_seq, mode='naive', scale=0.125))[0]
                o = projection(quantize(prefix + '/attn/GroupQueryAttention/output_0_QuantizeLinear', attention),
                               f'l{i}.o')
                gamma = ctx.weight(f'l{i}.ln2', p[f'l{i}.ln2'])
                h, residual = custom(prefix + '/post_attention_layernorm/SkipLayerNorm', [x, o, gamma])
                gu = projection(quantize(prefix + '/post_attention_layernorm/output_0_QuantizeLinear', h),
                                f'l{i}.gu')
                gate = b.extract_slice(gu, [0, 0], [t, INTER])
                up = b.extract_slice(gu, [0, INTER], [t, INTER])
                sigmoid = lowering.elementwise(b, imported_node(nodes, prefix + '/mlp/act_fn/Sigmoid', 'onnx.Sigmoid'),
                                               [gate])
                act = lowering.multiply(b, lowering.multiply(b, gate, sigmoid), up)
                imported_node(nodes, prefix + '/mlp/down_proj/MatMul_quant', 'onnx.MatMulInteger')
                imported_node(nodes, prefix + '/attn/o_proj/MatMul_quant', 'onnx.MatMulInteger')
                down = projection(quantize(prefix + '/mlp/Mul/output_0_QuantizeLinear', act), f'l{i}.down')
                gamma = ctx.weight(f'l{i}.next_norm', p[f'l{i}.next_norm'])
                next_norm = (f'/model/layers.{i + 1}/input_layernorm/SkipLayerNorm' if i + 1 < LAYERS
                             else f'/model/layers.{LAYERS}/final_norm_layernorm/SkipLayerNorm')
                h, x = custom(next_norm, [residual, down, gamma])
            last = b.extract_slice(h, [t - 1, 0], [1, HIDDEN])
            logits = b.matmul(last, head_value, b.fill(0.0, [1, VOCAB], 'f32'))
            b.store(logits, b.arg('logits'))
            b.emit('return')
            graph.entry_points[name] = ['weights', 'key', 'value', 'tokens', 'pos', 'logits']
        report, import_dir = src.provenance['import_report'], src.provenance['import_dir']
        graph.meta.update({'family': 'llama', 'source_format': 'onnx', 'source_sha256': src.provenance['sha256'],
            'state': [{'name': kind, 'shape': cache_shape, 'dtype': 'f32'} for kind in ('key', 'value')],
            'vocab': VOCAB, 'prefill_lengths': lengths, 'layers': layers,
            'integer_projections_per_forward': 4 * layers, 'original_integer_projections': 7 * layers,
            'options': {'max_seq': max_seq}, 'cpu_compiler': 'onnx-mlir-import/llvm-22',
            'onnx_import_sha256': report['artifact_sha256'],
            'fusions': q.meta['fusions'], 'onnx_quantization_unchanged': True})
        graph.files['tokenizer.json'] = src.root / 'tokenizer.json'
        graph.files['tokenizer_config.json'] = src.root / 'tokenizer_config.json'
        graph.files['compiler/onnx-import.json'] = import_dir / 'import.json'
        graph.files['compiler/onnx-graph.json'] = import_dir / 'graph.json'
        return graph


ARCH = SmolLMOnnx()
