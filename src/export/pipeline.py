"""The export front end as three stages, each runnable on its own:

    load(model, source)     -> SourceModel      read and verify; no numeric change
    quantize(src, recipe)   -> QuantizedModel   every array in its final numeric form
    to_mlir(q, shape)       -> ModelGraph       MLIR + weights; no file I/O, no quantization

``build_graph`` chains them for callers holding combined options
(LLMOptions, ResNetOptions); the compiler consumes its ModelGraph.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.export import model_arch
from src.export.ir.graph import ModelGraph
from src.export.ir.options import QuantRecipe, ShapeOptions
from src.export.ir.data import OnnxQuantWeight, QuantizedModel, QuantizedWeight, SourceModel
from src.export.load import detect_format
from src.export.quantize.plan import AwqHook


def architecture(model: str, source=None, source_format: str = 'auto'):
    """The architecture that reads ``model`` from ``source``, and the source format."""
    fmt = detect_format(source, source_format)
    arch = model_arch.get(model_arch.resolve(model, fmt))
    if source_format == 'auto' and fmt not in arch.formats and len(arch.formats) == 1:
        fmt = arch.formats[0]   # nothing to choose between: read the one format it has
    if fmt not in arch.formats:
        raise ValueError(f'Unsupported {arch.title or arch.name} source format: {fmt}')
    return arch, fmt


def split_options(opts: Any, prefill_lengths=(36, 40, 42)) -> tuple[QuantRecipe, ShapeOptions, int | None]:
    """Combined options -> (recipe, shape, layers)."""
    if hasattr(opts, 'split'):
        return opts.split(prefill_lengths)
    return QuantRecipe(bits=getattr(opts, 'bits', 8)), ShapeOptions(prefill_lengths=tuple(prefill_lengths)), None


def load(model: str, source=None, source_format: str = 'auto', layers: int | None = None, **kw) -> SourceModel:
    arch, fmt = architecture(model, source, source_format)
    return arch.load(str(source if source is not None else arch.default_source), fmt, layers, **kw)


def quantize(src: SourceModel, recipe: QuantRecipe, awq: AwqHook | None = None) -> QuantizedModel:
    return model_arch.get(src.arch).quantize(src, recipe, awq)


def to_mlir(q: QuantizedModel, shape: ShapeOptions) -> ModelGraph:
    return model_arch.get(q.source.arch).to_mlir(q, shape)


def build_graph(model: str, source=None, opts: Any = None, awq: AwqHook | None = None,
                source_format: str = 'auto', prefill_lengths=(36, 40, 42), onnx_import_dir=None,
                timings: dict[str, float] | None = None) -> tuple[ModelGraph, Any]:
    """Run load, quantize and to_mlir; return the graph and the parsed config."""
    arch, fmt = architecture(model, source, source_format)
    recipe, shape, layers = split_options(opts if opts is not None else arch.default_options(), prefill_lengths)
    arch.validate(recipe, shape, awq)
    times = {}
    start = time.monotonic()
    src = arch.load(str(source if source is not None else arch.default_source), fmt, layers,
                    onnx_import_dir=onnx_import_dir)
    times['load'] = time.monotonic() - start
    start = time.monotonic()
    q = arch.quantize(src, recipe, awq)
    times['quantize'] = time.monotonic() - start
    start = time.monotonic()
    graph = arch.to_mlir(q, shape)
    times['to_mlir'] = time.monotonic() - start
    if timings is not None:
        timings.update(times)
    return graph, q.config


# Stage results on disk, so each stage can run on its own.

QUANTIZED_FORMAT = 'npu-quantized/1'


def save_quantized(q: QuantizedModel, out: str | Path) -> Path:
    """``out/params.npz`` + ``out/model.json``; ``load_quantized`` restores it bit for bit."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrays, params = {}, {}
    for i, (name, value) in enumerate(q.params.items()):
        if isinstance(value, QuantizedWeight):
            params[name] = {'type': 'quantized', 'bits': value.bits, 'group': value.group}
            arrays[f'p{i}_q'], arrays[f'p{i}_scale'] = value.q, value.scale
        elif isinstance(value, OnnxQuantWeight):
            params[name] = {'type': 'onnx'}
            arrays[f'p{i}_q'], arrays[f'p{i}_scale'] = value.q, value.scale
        else:
            params[name] = {'type': 'array'}
            arrays[f'p{i}'] = np.asarray(value)
        params[name]['index'] = i
    np.savez(out / 'params.npz', **arrays)
    src = q.source
    paths = sorted(k for k, v in src.provenance.items() if isinstance(v, Path))
    model = {'format': QUANTIZED_FORMAT, 'arch': src.arch, 'source_format': src.format, 'source': src.source,
             'root': str(src.root), 'config': src.config, 'layers': src.layers,
             'provenance': {k: str(v) if k in paths else v for k, v in src.provenance.items()},
             'provenance_paths': paths, 'recipe': asdict(q.recipe), 'meta': q.meta, 'params': params}
    (out / 'model.json').write_text(json.dumps(model, indent=1))
    return out


def load_quantized(path: str | Path) -> QuantizedModel:
    path = Path(path)
    model = json.loads((path / 'model.json').read_text())
    if model.get('format') != QUANTIZED_FORMAT:
        raise ValueError(f'{path}: not a {QUANTIZED_FORMAT} directory')
    provenance = {k: Path(v) if k in model['provenance_paths'] else v for k, v in model['provenance'].items()}
    src = SourceModel(model['arch'], model['source_format'], model['source'], Path(model['root']), model['config'],
                      None, model['layers'], provenance)
    arch = model_arch.get(src.arch)
    params = {}
    with np.load(path / 'params.npz') as arrays:
        for name, info in model['params'].items():
            i = info['index']
            if info['type'] == 'quantized':
                params[name] = QuantizedWeight(arrays[f'p{i}_q'], arrays[f'p{i}_scale'], info['bits'], info['group'])
            elif info['type'] == 'onnx':
                params[name] = OnnxQuantWeight(arrays[f'p{i}_q'], arrays[f'p{i}_scale'])
            else:
                params[name] = arrays[f'p{i}']
    return QuantizedModel(src, QuantRecipe(**model['recipe']), arch.config(src), params, model['meta'])


def save_graph(graph: ModelGraph, out: str | Path) -> Path:
    """``out/model.mlir``, ``out/weights.npz`` and ``out/graph.json`` (name,
    entry points, meta, files): everything the compiler reads from a graph."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'model.mlir').write_text(graph.module.render())
    np.savez(out / 'weights.npz', **{f'w{i}': a for i, a in enumerate(graph.weights.values())})
    (out / 'graph.json').write_text(json.dumps({
        'name': graph.name, 'weights': list(graph.weights), 'entry_points': graph.entry_points,
        'meta': graph.meta, 'files': {k: str(v) for k, v in graph.files.items()}}, indent=1, default=str))
    return out


# ---------------------------------------------------------------------------
# Export through PyTorch (docs/plans/2026-1029-002-export-pytorch-exportedprogram.md):
#
#   load (framework + torchao quantization) -> export_graph (torch.export)
#   -> to_mlir (torch-mlir: model.mlir, linalg) -> npu-compile (C++ back end)
#
# The front end runs wherever torch does; npu-compile needs Linux with LLVM 22.
# On Windows it runs in WSL (paths are translated).

PACKAGE_FORMAT = 'npu-package/2'
REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / 'src' / 'export' / 'backend'
BACKEND_SOURCES = (BACKEND, REPO / 'src' / 'vendors' / 'npu_pynq' / 'custom_dialect')


def read_config(path: str | Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def _linux_path(path: Path) -> str:
    import sys
    if not sys.platform.startswith('win'):
        return str(path)
    p = Path(path).resolve()
    return f'/mnt/{p.drive[0].lower()}' + p.as_posix()[2:]


def _linux(command: str) -> None:
    import subprocess
    import sys
    args = ['wsl.exe', '-e', 'bash', '-lc', command] if sys.platform.startswith('win') else ['bash', '-lc', command]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f'{command.split()[0]} failed:\n{proc.stdout[-3000:]}{proc.stderr[-3000:]}')


def build_backend(build_dir: str | None = None, llvm: str = '/usr/lib/llvm-22') -> str:
    """Build npu-compile with CMake (incrementally); return its Linux path."""
    import os
    build = build_dir or os.environ.get('NPU_BACKEND_BUILD') or _linux_path(REPO / 'build' / 'npu-compile')
    _linux(f"cmake -S '{_linux_path(BACKEND)}' -B {build} -G Ninja -DCMAKE_BUILD_TYPE=Release "
           f"-DMLIR_DIR={llvm}/lib/cmake/mlir >/dev/null && cmake --build {build}")
    return f'{build}/npu-compile'


def export_package(config: dict, out: str | Path, source: str | None = None,
                   llvm_bin: str = '/usr/lib/llvm-22/bin', timings: dict | None = None) -> Path:
    """Run the whole export for ``config`` (a configs/*.yml); write the package to ``out``."""
    import shutil
    from src.export.frontend.export_graph import export_llm
    from src.export.frontend.load import framework, load as load_model
    from src.export.frontend.to_mlir import to_mlir
    from src.vendors.npu_pynq.isa import isa

    times = timings if timings is not None else {}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = config['load']
    src = source or spec['source']
    if not Path(src).is_absolute() and (REPO / src).exists():
        src = str(REPO / src)
    start = time.monotonic()
    model = load_model(spec['framework'], src, quantization=spec.get('quantization'), require=spec.get('require'))
    times['load'] = time.monotonic() - start
    graph = config['graph']
    if graph['kind'] != 'llm':
        raise ValueError(f"unsupported graph kind {graph['kind']!r}")
    start = time.monotonic()
    programs, cache = export_llm(model, graph['max_seq'], graph['chunk'])
    times['export_graph'] = time.monotonic() - start
    start = time.monotonic()
    model_mlir = to_mlir(programs, out / 'ir')
    times['to_mlir'] = time.monotonic() - start

    start = time.monotonic()
    tool = build_backend()
    times['build_backend'] = time.monotonic() - start
    limits = isa.Limits()
    start = time.monotonic()
    _linux(f"{tool} '{_linux_path(model_mlir)}' --out '{_linux_path(out)}' "
           f"--devices {','.join(config['devices'])} --cpus {','.join(config['cpus'])} "
           f"--llvm-bin {llvm_bin} --rows {limits.rows} --columns {limits.columns} --max-k {limits.max_k}")
    times['backend'] = time.monotonic() - start

    compiled = json.loads((out / 'compile.json').read_text())
    names = cache.names()
    compiled['entry_points']['decode']['names'] = {'inputs': ['tokens', 'pos', *names],
                                                   'outputs': ['logits', *names]}
    compiled['entry_points']['prefill']['names'] = {'inputs': ['tokens', 'pos', 'last', *names],
                                                    'outputs': ['logits', *names]}
    for name, path in framework(spec['framework']).files(src).items():
        shutil.copy2(path, out / name)
    eos = model.config.eos_token_id
    manifest = {**compiled, 'format': PACKAGE_FORMAT, 'model': config['model'],
                'llm': {'vocab': model.config.vocab_size, 'max_seq': graph['max_seq'], 'chunk': graph['chunk'],
                        'cache': {'layers': cache.layers, 'shape': list(cache.shape)},
                        'eos': eos if isinstance(eos, list) else [eos]},
                'quantization': spec.get('quantization'), 'compile_seconds': times}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return out
