"""Build with unmodified src; optional MLIR/NumPy interchange across hosts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

REPO = Path(__file__).resolve().parents[3]
EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('stage', 'compile', 'build'))
    parser.add_argument('--source', type=Path, default=EXAMPLE / 'model/checkpoint')
    parser.add_argument('--stage', type=Path, default=EXAMPLE / 'model/graph')
    parser.add_argument('--out', type=Path, default=REPO / 'dist/smollm-safetensors/package')
    parser.add_argument('--targets', default='host,pynq')
    parser.add_argument('--attention', choices=('online', 'naive'), default='online')
    args = parser.parse_args()
    if args.phase in ('stage', 'build'):
        from src.export.graphs.smollm.model import build
        from src.export.core.graph import LLMOptions
        spec = json.loads((EXAMPLE / 'model-source.json').read_text())
        for name, info in spec['files'].items():
            with (args.source / name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != info['sha256']:
                    raise ValueError('Source identity mismatch: ' + name)
        graph, _ = build(str(args.source.resolve()), LLMOptions(attention=args.attention), source_format='safetensors')
        graph.files['tokenizer_config.json'] = args.source / 'tokenizer_config.json'
        if args.phase == 'stage':
            args.stage.mkdir(parents=True, exist_ok=True)
            (args.stage / 'model.mlir').write_text(graph.module.render(), encoding='utf-8')
            np.savez(args.stage / 'weights.npz', **graph.weights)
            (args.stage / 'graph.json').write_text(json.dumps({
                'name': graph.name, 'entry_points': graph.entry_points, 'meta': graph.meta,
                'source_sha256': spec['files']['model.safetensors']['sha256'],
            }, indent=2) + '\n', encoding='utf-8')
            print('Staged all', graph.meta['config']['layers'], 'layers', flush=True)
            return
    else:
        from src.export.core.graph import ModelGraph
        data = json.loads((args.stage / 'graph.json').read_text())
        text = (args.stage / 'model.mlir').read_text()
        with np.load(args.stage / 'weights.npz', allow_pickle=False) as archive:
            weights = {name: archive[name] for name in archive.files}
        graph = ModelGraph(data['name'], SimpleNamespace(render=lambda: text), weights,
                           data['entry_points'], data['meta'],
                           {name: args.source / name for name in ('tokenizer.json', 'tokenizer_config.json')})
    from src.export.cli import export
    result = export(graph, args.out, targets=args.targets.split(','))
    print((result / 'manifest.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
