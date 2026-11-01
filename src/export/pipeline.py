"""The export, from a config to a package:

    load           a framework builds the model; our torchao config quantizes it
                   (frontend/load, frontend/quantization)
    export_graph   torch.export: one ExportedProgram per entry point
    to_mlir        torch-mlir: model.mlir, linalg on tensors
    npu-compile    the C++ back end (backend/): delegates, weights.bin, CPU code

The front end runs wherever torch does; npu-compile needs Linux with LLVM 22.
On Windows it runs in WSL (paths are translated). See
docs/plans/2026-1029-002-export-pytorch-exportedprogram.md.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

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
    from src.export.frontend.export_graph import export_classifier, export_llm
    from src.export.frontend.load import framework, load as load_model
    from src.export.frontend.to_mlir import to_mlir
    from src.vendors.npu_pynq.spec import isa

    times = timings if timings is not None else {}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = config['load']
    src = source or spec['source']
    if not Path(src).is_absolute() and (REPO / src).exists():
        src = str(REPO / src)
    start = time.monotonic()
    extra = {k: v for k, v in spec.items() if k not in ('framework', 'source', 'quantization', 'require')}
    model = load_model(spec['framework'], src, quantization=spec.get('quantization'), require=spec.get('require'),
                       **extra)
    times['load'] = time.monotonic() - start
    graph = config['graph']
    start = time.monotonic()
    if graph['kind'] == 'llm':
        programs, cache = export_llm(model, graph['max_seq'], graph['chunk'])
    elif graph['kind'] == 'classifier':
        programs, cache = export_classifier(model, tuple(graph['input']['shape'])), None
    else:
        raise ValueError(f"unsupported graph kind {graph['kind']!r}")
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
    for name, path in framework(spec['framework']).files(src).items():
        shutil.copy2(path, out / name)
    manifest = {**compiled, 'format': PACKAGE_FORMAT, 'model': config['model'],
                'quantization': spec.get('quantization'), 'compile_seconds': times}
    if graph['kind'] == 'llm':
        names = cache.names()
        compiled['entry_points']['decode']['names'] = {'inputs': ['tokens', 'pos', *names],
                                                       'outputs': ['logits', *names]}
        compiled['entry_points']['prefill']['names'] = {'inputs': ['tokens', 'pos', 'last', *names],
                                                        'outputs': ['logits', *names]}
        eos = model.config.eos_token_id
        manifest['llm'] = {'vocab': model.config.vocab_size, 'max_seq': graph['max_seq'], 'chunk': graph['chunk'],
                           'cache': {'layers': cache.layers, 'shape': list(cache.shape)},
                           'eos': eos if isinstance(eos, list) else [eos]}
    else:
        compiled['entry_points']['forward']['names'] = {'inputs': ['image'], 'outputs': ['logits']}
        manifest['classifier'] = {'input': graph['input'],
                                  'classes': compiled['entry_points']['forward']['outputs'][0]['shape'][-1]}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return out
