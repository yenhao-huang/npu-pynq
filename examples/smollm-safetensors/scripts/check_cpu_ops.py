"""Isolate shared CPU operators across host and ARM without editing src."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def stage(root, checkpoint=None):
    from src.export.core.graph import ModelGraph
    from src.export.compilers.mlir.builder import ModuleBuilder, memref_type
    from src.export.compilers.mlir.ops.context import Ctx
    from src.export.compilers.mlir.ops import core, attention as A, embedding as E, linear as L
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 576)).astype(np.float32)
    q = rng.normal(size=(2, 576)).astype(np.float32)
    k = rng.normal(size=(2, 192)).astype(np.float32)
    v = rng.normal(size=(2, 192)).astype(np.float32)
    mod = ModuleBuilder()
    args = [('ws', 'memref<?xi8>')]
    args += [(name, memref_type(shape, 'f32')) for name, shape in
             [('x', x.shape), ('q', q.shape), ('k', k.shape), ('v', v.shape), ('qkv', (2, 960)),
              ('kc', (1, 3, 8, 64)), ('vc', (1, 3, 8, 64)),
              ('norm', x.shape), ('silu', x.shape), ('online', x.shape), ('naive', x.shape),
              ('rope', x.shape), ('mlp', x.shape)]]
    if checkpoint:
        args += [('tokens', memref_type([2], 'i32')), ('embedding', memref_type([2,576], 'f32')),
                 ('logits', memref_type([2,49152], 'f32'))]
    f = mod.func('forward', args)
    graph = ModelGraph('cpu-operators', mod)
    ctx = Ctx(f, graph, f.arg('ws'))
    xt = f.to_tensor(f.arg('x'))
    f.store(core.rmsnorm(f, xt, ctx.weight('norm', np.ones(576, np.float32)), 1e-5), f.arg('norm'))
    f.store(core.silu(f, xt), f.arg('silu'))
    expected = {'norm': x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-5),
                'silu': x / (1 + np.exp(-x))}
    cos, sin = E.rope_tables(512, 64, 10000.)
    rotated = E.rope(ctx, xt, 9, 64, f.index(3), ctx.weight('cos', cos), ctx.weight('sin', sin))
    f.store(rotated, f.arg('rope'))
    expected['rope'] = E.ref_rope(x, 9, 64, 3, cos, sin)
    gate = L.quantize_weight((rng.normal(size=(3072,576))*.02).astype(np.float32), 8)
    down = L.quantize_weight((rng.normal(size=(576,1536))*.02).astype(np.float32), 8)
    gu = L.linear(ctx, xt, 'gate_up', gate)
    act = core.silu_mul(f, f.extract_slice(gu,[0,0],[2,1536]), f.extract_slice(gu,[0,1536],[2,1536]))
    f.store(L.linear(ctx, act, 'down', down), f.arg('mlp'))
    gu_np = L.ref_linear(x, gate)
    act_np = gu_np[:,:1536] / (1 + np.exp(-gu_np[:,:1536])) * gu_np[:,1536:]
    expected['mlp'] = L.ref_linear(act_np, down)
    for mode in ('online', 'naive'):
        cfg = A.AttnConfig(2, 9, 3, 64, 1, 8, mode, 'f32')
        qkv = f.to_tensor(f.arg('qkv'))
        out = A.attention(ctx, cfg, f.extract_slice(qkv, [0,0], [2,576]),
                          f.extract_slice(qkv, [0,576], [2,192]),
                          f.extract_slice(qkv, [0,768], [2,192]), {'k': f.arg('kc'), 'v': f.arg('vc')},
                          f.index(0), f.index(0))
        f.store(out, f.arg(mode))
        cache = {n: np.zeros(s, np.float32) for n, (s, _) in cfg.cache_shapes().items()}
        expected[mode] = A.ref_attention(cfg, q, k, v, cache, 0, 0)
    if checkpoint:
        from src.export.compilers.mlir.ops import embedding as E, linear as L
        from src.export.core.model_loader import SafeTensors
        from src.vendors.npu_pynq.isa import isa
        with SafeTensors(checkpoint) as tensors:
            weight = L.quantize_weight(tensors['model.embed_tokens.weight'], 8, 128)
        rows = E.tied_lookup(ctx, f.to_tensor(f.arg('tokens')), 'head', weight, isa.Limits())
        f.store(rows, f.arg('embedding'))
        f.store(L.linear(ctx, rows, 'head', weight), f.arg('logits'))
        token_ids = np.array([1,4093], np.int32)
        expected['embedding'] = weight.q[:, token_ids].T.astype(np.float32) * weight.scale[token_ids,None]
        expected['logits'] = L.ref_linear(expected['embedding'], weight)
    f.emit('return')
    (root / 'model.mlir').write_text(mod.render())
    np.savez(root / 'weights.npz', **graph.weights)
    (root / 'graph.json').write_text(json.dumps({'name':graph.name, 'meta':graph.meta,
                                               'entry_points':{'forward':[n for n, _ in args]}}))
    np.savez(root / 'inputs.npz', x=x, q=q, k=k, v=v)
    np.savez(root / 'expected.npz', **expected)


def run(root, package, target):
    from src.runtime.core.package import Package
    model = Package(package, backend='sim', target=target)
    with np.load(root / 'inputs.npz') as data:
        x, q, k, v = [data[n].copy() for n in ('x', 'q', 'k', 'v')]
    kc, vc = [np.zeros((1,3,8,64), np.float32) for _ in range(2)]
    outputs = {n:np.zeros_like(x) for n in ('norm', 'silu', 'online', 'naive', 'rope', 'mlp')}
    qkv = np.concatenate((q, k, v), axis=1)
    args = ['weights', x, q, k, v, qkv, kc, vc, *outputs.values()]
    with np.load(root / 'expected.npz') as expected:
        if 'embedding' in expected.files:
            for name in ('embedding', 'logits'):
                outputs[name] = np.zeros_like(expected[name])
            args += [np.array([1,4093], np.int32), outputs['embedding'], outputs['logits']]
    model.call('forward', *args)
    result = {}
    with np.load(root / 'expected.npz') as expected:
        for name, actual in outputs.items():
            result[name] = {'max_abs_error':float(np.max(np.abs(actual - expected[name]))),
                            'matches':bool(np.allclose(actual, expected[name], rtol=2e-4, atol=2e-4))}
    print(json.dumps(result, indent=2))
    (root / (target + '-result.json')).write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('stage', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--package', type=Path)
    parser.add_argument('--target', default='host')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    if args.phase == 'stage':
        stage(args.root, args.checkpoint)
    else:
        run(args.root, args.package, args.target)


if __name__ == '__main__':
    main()
