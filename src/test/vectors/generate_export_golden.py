"""Golden digests of every export frontend's ModelGraph.

Each case writes a tiny seeded checkpoint, runs the frontend, and records the
SHA-256 of the rendered MLIR, of every weight (name, dtype, shape, bytes), of
``meta`` and of ``entry_points``. The digests pin the front end bit for bit
while it is restructured (docs/plans/2026-1029-001-export-load-quantize-to-mlir.md).

    python -m src.test.vectors.generate_export_golden           # rewrite the file
    python -m src.test.vectors.generate_export_golden --check   # compare only

The SmolLM ONNX case needs the pinned ``model_int8.onnx`` and a verified
ONNX-MLIR import directory holding ``graph.json``; set NPU_GOLDEN_ONNX and
NPU_GOLDEN_ONNX_IMPORT to enable it. ResNet-18 runs without PyTorch through a
stand-in ``torch.load``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
from unittest import mock

import numpy as np

GOLDEN = Path(__file__).with_name('export_golden.json')


def write_safetensors(path: Path, arrays: dict[str, np.ndarray]) -> None:
    header, blocks, offset = {}, [], 0
    for name, array in arrays.items():
        block = np.ascontiguousarray(array, np.float32).tobytes()
        header[name] = {'dtype': 'F32', 'shape': list(array.shape), 'data_offsets': [offset, offset + len(block)]}
        blocks.append(block)
        offset += len(block)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + b''.join(blocks))


def write_checkpoint(root: Path, config: dict, shapes: dict[str, tuple[int, ...]], seed: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arrays = {}
    for name, shape in shapes.items():
        if name.endswith('norm.weight'):
            arrays[name] = (1.0 + 0.1 * rng.normal(size=shape)).astype(np.float32)
        else:
            arrays[name] = (0.1 * rng.normal(size=shape)).astype(np.float32)
    write_safetensors(root / 'model.safetensors', arrays)
    (root / 'config.json').write_text(json.dumps(config))
    (root / 'tokenizer.json').write_text('{}')
    return root


def llama_shapes(hidden, inter, heads, kv_heads, head_dim, layers, vocab, qk_norm=False, tied=True):
    shapes = {'model.embed_tokens.weight': (vocab, hidden), 'model.norm.weight': (hidden,)}
    if not tied:
        shapes['lm_head.weight'] = (vocab, hidden)
    for i in range(layers):
        p = f'model.layers.{i}.'
        shapes.update({p + 'self_attn.q_proj.weight': (heads * head_dim, hidden),
                       p + 'self_attn.k_proj.weight': (kv_heads * head_dim, hidden),
                       p + 'self_attn.v_proj.weight': (kv_heads * head_dim, hidden),
                       p + 'self_attn.o_proj.weight': (hidden, heads * head_dim),
                       p + 'mlp.gate_proj.weight': (inter, hidden),
                       p + 'mlp.up_proj.weight': (inter, hidden),
                       p + 'mlp.down_proj.weight': (hidden, inter),
                       p + 'input_layernorm.weight': (hidden,),
                       p + 'post_attention_layernorm.weight': (hidden,)})
        if qk_norm:
            shapes[p + 'self_attn.q_norm.weight'] = (head_dim,)
            shapes[p + 'self_attn.k_norm.weight'] = (head_dim,)
    return shapes


def smollm_source(root: Path) -> Path:
    config = dict(hidden_size=16, intermediate_size=32, num_attention_heads=2, num_key_value_heads=1,
                  num_hidden_layers=2, rms_norm_eps=1e-5, vocab_size=32, tie_word_embeddings=True,
                  eos_token_id=2)
    return write_checkpoint(root, config, llama_shapes(16, 32, 2, 1, 8, 2, 32), 1)


def qwen3_source(root: Path) -> Path:
    config = dict(hidden_size=16, intermediate_size=32, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=8, num_hidden_layers=2, rms_norm_eps=1e-6, rope_theta=1000000.0, vocab_size=32,
                  eos_token_id=2)
    return write_checkpoint(root, config, llama_shapes(16, 32, 4, 2, 8, 2, 32, qk_norm=True), 2)


def qwen3_moe_source(root: Path) -> Path:
    config = dict(hidden_size=16, intermediate_size=32, moe_intermediate_size=16, num_experts=4,
                  num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[1],
                  num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_hidden_layers=2,
                  rms_norm_eps=1e-6, rope_theta=1000000.0, vocab_size=32, tie_word_embeddings=False,
                  eos_token_id=2)
    shapes = llama_shapes(16, 32, 2, 1, 8, 2, 32, qk_norm=True, tied=False)
    p = 'model.layers.0.'
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        del shapes[p + 'mlp.' + name + '.weight']
    shapes[p + 'mlp.gate.weight'] = (4, 16)
    for e in range(4):
        q = f'{p}mlp.experts.{e}.'
        shapes.update({q + 'gate_proj.weight': (16, 16), q + 'up_proj.weight': (16, 16),
                       q + 'down_proj.weight': (16, 16)})
    return write_checkpoint(root, config, shapes, 3)


def qwen3_5_source(root: Path) -> Path:
    hidden, inter, heads, kv_heads, head_dim = 16, 32, 2, 1, 8
    k_heads, v_heads, dk, dv, kernel = 1, 2, 8, 8, 4
    config = dict(hidden_size=hidden, intermediate_size=inter, num_attention_heads=heads,
                  num_key_value_heads=kv_heads, head_dim=head_dim, num_hidden_layers=2,
                  layer_types=['linear_attention', 'full_attention'], rms_norm_eps=1e-6,
                  rope_parameters={'rope_theta': 10000000.0, 'partial_rotary_factor': 0.25},
                  linear_num_key_heads=k_heads, linear_num_value_heads=v_heads, linear_key_head_dim=dk,
                  linear_value_head_dim=dv, linear_conv_kernel_dim=kernel, vocab_size=32, eos_token_id=2)
    pre = 'model.language_model.'
    channels = 2 * k_heads * dk + v_heads * dv
    shapes = {pre + 'embed_tokens.weight': (32, hidden), pre + 'norm.weight': (hidden,)}
    for i, kind in enumerate(config['layer_types']):
        p = f'{pre}layers.{i}.'
        shapes.update({p + 'input_layernorm.weight': (hidden,), p + 'post_attention_layernorm.weight': (hidden,),
                       p + 'mlp.gate_proj.weight': (inter, hidden), p + 'mlp.up_proj.weight': (inter, hidden),
                       p + 'mlp.down_proj.weight': (hidden, inter)})
        if kind == 'full_attention':
            shapes.update({p + 'self_attn.q_proj.weight': (2 * heads * head_dim, hidden),
                           p + 'self_attn.k_proj.weight': (kv_heads * head_dim, hidden),
                           p + 'self_attn.v_proj.weight': (kv_heads * head_dim, hidden),
                           p + 'self_attn.o_proj.weight': (hidden, heads * head_dim),
                           p + 'self_attn.q_norm.weight': (head_dim,), p + 'self_attn.k_norm.weight': (head_dim,)})
        else:
            la = p + 'linear_attn.'
            shapes.update({la + 'in_proj_qkv.weight': (channels, hidden),
                           la + 'in_proj_z.weight': (v_heads * dv, hidden),
                           la + 'in_proj_b.weight': (v_heads, hidden), la + 'in_proj_a.weight': (v_heads, hidden),
                           la + 'out_proj.weight': (hidden, v_heads * dv), la + 'conv1d.weight': (channels, 1, kernel),
                           la + 'A_log': (v_heads,), la + 'dt_bias': (v_heads,), la + 'norm.weight': (dv,)})
    return write_checkpoint(root, config, shapes, 4)


def resnet18_state(seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = {}

    def conv(name, out_c, in_c, k):
        state[name + '.weight'] = (rng.normal(size=(out_c, in_c, k, k)) * (2.0 / (in_c * k * k)) ** 0.5)

    def bn(name, c):
        state[name + '.weight'] = 1.0 + 0.1 * rng.normal(size=c)
        state[name + '.bias'] = 0.1 * rng.normal(size=c)
        state[name + '.running_mean'] = 0.1 * rng.normal(size=c)
        state[name + '.running_var'] = rng.uniform(0.5, 1.5, size=c)

    conv('conv1', 64, 3, 7)
    bn('bn1', 64)
    in_c = 64
    for stage, out_c in enumerate((64, 128, 256, 512), start=1):
        for block in range(2):
            p = f'layer{stage}.{block}.'
            conv(p + 'conv1', out_c, in_c if block == 0 else out_c, 3)
            bn(p + 'bn1', out_c)
            conv(p + 'conv2', out_c, out_c, 3)
            bn(p + 'bn2', out_c)
            if block == 0 and stage > 1:
                conv(p + 'downsample.0', out_c, in_c, 1)
                bn(p + 'downsample.1', out_c)
        in_c = out_c
    state['fc.weight'] = rng.normal(size=(1000, 512)) * 0.05
    state['fc.bias'] = rng.normal(size=1000) * 0.05
    return {k: v.astype(np.float32) for k, v in state.items()}


def fake_torch(state: dict[str, np.ndarray]) -> types.ModuleType:
    """A ``torch`` module whose ``load`` returns ``state`` as detachable tensors."""
    class Tensor:
        def __init__(self, array):
            self.array = array

        def detach(self):
            return self

        def numpy(self):
            return self.array

    module = types.ModuleType('torch')
    module.load = lambda *_args, **_kwargs: {k: Tensor(v) for k, v in state.items()}
    return module


def resnet18_source(root: Path) -> Path:
    path = root / 'resnet18.pth'
    path.write_bytes(b'stand-in checkpoint')
    return path


def onnx_source(root: Path) -> Path | None:
    onnx = os.environ.get('NPU_GOLDEN_ONNX')
    if not onnx or not Path(onnx).is_file():
        return None
    target = root / 'model_int8.onnx'
    try:
        os.link(onnx, target)
    except OSError:
        import shutil
        shutil.copy2(onnx, target)
    (root / 'config.json').write_text(json.dumps(dict(
        hidden_size=576, intermediate_size=1536, num_attention_heads=9, num_key_value_heads=3,
        num_hidden_layers=30, rms_norm_eps=1e-5, rope_theta=100000, vocab_size=49152,
        tie_word_embeddings=True, eos_token_id=2)))
    return target


def llm_options(**kw):
    from src.export.ir.options import LLMOptions
    base = dict(chunk=3, max_seq=16, group=8, attention='naive')
    base.update(kw)
    return LLMOptions(**base)


def resnet_options(bits):
    from src.export.model_arch.resnet18.arch import ResNetOptions
    return ResNetOptions(bits=bits)


# name -> (model, source writer, options factory, extra build kwargs)
CASES = {
    'smollm-w8': ('smollm', smollm_source, lambda: llm_options(), {}),
    'smollm-w4-int8kv-online': ('smollm', smollm_source,
                                lambda: llm_options(bits=4, kv='int8', attention='online', layers=1), {}),
    'smollm-fp32': ('smollm', smollm_source, lambda: llm_options(bits=32), {}),
    'qwen3-w8': ('qwen3', qwen3_source, lambda: llm_options(), {}),
    'qwen3-w4': ('qwen3', qwen3_source, lambda: llm_options(bits=4, embed_bits=4), {}),
    'qwen3_moe-w8': ('qwen3_moe', qwen3_moe_source, lambda: llm_options(), {}),
    'qwen3_5-w8': ('qwen3_5', qwen3_5_source, lambda: llm_options(), {}),
    'resnet18-w8': ('resnet18', resnet18_source, lambda: resnet_options(8), {}),
    'resnet18-fp32': ('resnet18', resnet18_source, lambda: resnet_options(32), {}),
    'smollm-onnx': ('smollm', onnx_source, lambda: llm_options(chunk=16, max_seq=512, group=128,
                                                               attention='online', layers=2),
                    {'prefill_lengths': (4,)}),
}


def frontend_graph(model: str, source: Path, opts, **kw):
    """The ModelGraph the front end (load, quantize, to_mlir) builds for ``source``."""
    from src.export.pipeline import build_graph
    graph, _ = build_graph(model, str(source), opts, **kw)
    return graph


def digest(graph, root: Path) -> dict[str, str]:
    def text(value) -> str:
        out = json.dumps(value, sort_keys=True, default=str)
        for form in (str(root), root.as_posix(), json.dumps(str(root))[1:-1]):
            out = out.replace(form, '<root>')
        return out

    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    weights = hashlib.sha256()
    for name in sorted(graph.weights):
        array = np.ascontiguousarray(graph.weights[name])
        weights.update(f'{name}|{array.dtype.str}|{array.shape}|'.encode())
        weights.update(array.tobytes())
    return {'module': sha(graph.module.render().encode()),
            'weights': weights.hexdigest(),
            'weight_count': str(len(graph.weights)),
            'meta': sha(text(graph.meta).encode()),
            'entry_points': sha(text(graph.entry_points).encode()),
            'files': text(sorted(graph.files))}


def run_case(name: str) -> dict[str, str] | None:
    """Digests for one case, or None when its inputs are unavailable."""
    model, writer, options, kw = CASES[name]
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        root = Path(tmp).resolve()
        source = writer(root)
        if source is None:
            return None
        kw = dict(kw)
        if model == 'resnet18':
            stack.enter_context(mock.patch.dict(sys.modules, {'torch': fake_torch(resnet18_state())}))
        if name == 'smollm-onnx':
            imported = os.environ.get('NPU_GOLDEN_ONNX_IMPORT')
            if not imported or not (Path(imported) / 'graph.json').is_file():
                return None
            graph_json = Path(imported) / 'graph.json'
            stack.enter_context(mock.patch(
                'src.export.compilers.onnx_mlir.adapter.inspect_ir',
                lambda _module, _output: json.loads(graph_json.read_text(encoding='utf-8'))))
            kw['onnx_import_dir'] = imported
        graph = frontend_graph(model, source, options(), **kw)
        result = digest(graph, root)
        graph.weights.clear()
        return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='compare with the recorded digests')
    parser.add_argument('cases', nargs='*', default=list(CASES))
    args = parser.parse_args(argv)
    recorded = json.loads(GOLDEN.read_text()) if GOLDEN.is_file() else {}
    failed = False
    for name in args.cases:
        result = run_case(name)
        if result is None:
            print(f'{name}: skipped (inputs unavailable)')
            continue
        if args.check:
            ok = recorded.get(name) == result
            failed |= not ok
            print(f'{name}: {"ok" if ok else "MISMATCH"}')
        else:
            recorded[name] = result
            print(f'{name}: recorded')
    if not args.check:
        GOLDEN.write_text(json.dumps(recorded, indent=2, sort_keys=True) + '\n')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
