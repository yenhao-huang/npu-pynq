"""Stage the Torch frontend on Windows; compile the same graph on Linux.

The interchange is plain MLIR, JSON and NumPy arrays, never pickle. This keeps
the experiment independent of a second PyTorch installation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('stage','compile'))
    p.add_argument('--source', type=Path, default=ROOT/'examples/resnet18/model/resnet18-f37072fd.pth')
    p.add_argument('--stage', type=Path, default=ROOT/'examples/resnet18/model/aot-source')
    p.add_argument('--out', type=Path, default=ROOT/'examples/resnet18/model/aot')
    p.add_argument('--targets', default='host,pynq')
    args = p.parse_args()
    if args.phase == 'stage':
        from src.export.graphs.resnet18.model import build
        started = time.perf_counter()
        graph, _ = build(str(args.source))
        args.stage.mkdir(parents=True, exist_ok=True)
        (args.stage/'model.mlir').write_text(graph.module.render(),encoding='utf-8')
        np.savez(args.stage/'weights.npz', **graph.weights)
        meta = {'name':graph.name, 'entry_points':graph.entry_points, 'meta':graph.meta,
                'checkpoint_sha256':hashlib.file_digest(args.source.open('rb'),'sha256').hexdigest(),
                'frontend_seconds':time.perf_counter()-started}
        (args.stage/'graph.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(meta,indent=2))
    else:
        from src.export.cli import export
        from src.export.core.graph import ModelGraph
        meta = json.loads((args.stage/'graph.json').read_text())
        text = (args.stage/'model.mlir').read_text()
        with np.load(args.stage/'weights.npz',allow_pickle=False) as archive:
            weights = {name:archive[name] for name in archive.files}
        graph = ModelGraph(meta['name'], SimpleNamespace(render=lambda:text), weights,
                           meta['entry_points'], meta['meta'])
        result = export(graph,args.out,targets=args.targets.split(','))
        print((result/'manifest.json').read_text(),flush=True)


if __name__ == '__main__': main()
