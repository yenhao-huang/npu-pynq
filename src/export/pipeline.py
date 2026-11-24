"""The export, from a config to a package:

    load           a framework builds the float model (frontend/load)
    export_graph   torch.export: one ExportedProgram per entry point, undecomposed
    quantize       a graph pass quantizes each program (frontend/quantization),
                   calibrated on the config's calibration: inputs
    to_mlir        ATen decompositions, then torch-mlir's importer: torch.mlir, the torch dialect
    pynqo-middle-end the C++ middle end (middle-end/): our rewrites, torch-mlir's
                   pipelines to linalg on tensors, then device assignment and
                   partitioning: model.mlir (the CPU graph, with delegate calls)
                   and ir/delegates/<device>.mlir (each device's subgraphs)
    pynqo-backend    the C++ back end (backend/): each delegate compiles its device
                   module, weights.bin, CPU code

The front end runs wherever torch does; both C++ tools need Linux with LLVM 22.
On Windows they run in WSL (paths are translated). See
docs/plans/2026-1029-002-export-pytorch-exportedprogram.md.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

# The runtime refuses a package with any other format.
# Keep this equal to FORMAT in src/runtime/share/loader.py.
PACKAGE_FORMAT = 'pynqo-package/3'
REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / 'src' / 'export' / 'backend'
DEVICES = REPO / 'src' / 'export' / 'devices'
BACKEND_SOURCES = (BACKEND, DEVICES / 'npu', DEVICES / 'risc_v_npu')
MIDDLE_END = REPO / 'src' / 'export' / 'middle-end'


def read_config(path: str | Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def _linux_path(path: Path) -> str:
    """Return ``path`` as the C++ tools see it.

    On Windows they run inside WSL, so C:\\x becomes /mnt/c/x.
    """
    import sys
    if not sys.platform.startswith('win'):
        return str(path)
    p = Path(path).resolve()
    return f'/mnt/{p.drive[0].lower()}' + p.as_posix()[2:]


def _linux(command: str) -> None:
    import subprocess
    import sys
    # -l starts a login shell, so the command gets the PATH from the user's profile.
    args = ['wsl.exe', '-e', 'bash', '-lc', command] if sys.platform.startswith('win') else ['bash', '-lc', command]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f'{command.split()[0]} failed:\n{proc.stdout[-3000:]}{proc.stderr[-3000:]}')


def backend_tool() -> str:
    """The Linux path of pynqo-backend, built or not."""
    import os
    build = os.environ.get('PYNQO_BACKEND_BUILD') or _linux_path(REPO / 'build' / 'pynqo-backend')
    return f'{build}/pynqo-backend'


def build_backend(build_dir: str | None = None, llvm: str = '/usr/lib/llvm-22') -> str:
    """Build pynqo-backend with CMake (incrementally); return its Linux path.

    Every export calls this, so a package is never compiled by a stale pynqo-backend.
    The call is cheap: Ninja does nothing when the sources are unchanged.
    """
    build = build_dir or backend_tool().rsplit('/', 1)[0]
    _linux(f"cmake -S '{_linux_path(BACKEND)}' -B {build} -G Ninja -DCMAKE_BUILD_TYPE=Release "
           f"-DMLIR_DIR={llvm}/lib/cmake/mlir >/dev/null && cmake --build {build}")
    return f'{build}/pynqo-backend'


def middle_end_tool() -> str:
    """The Linux path of pynqo-middle-end, built or not."""
    import os
    build = os.environ.get('PYNQO_MIDDLE_END_BUILD') or _linux_path(REPO / 'build' / 'pynqo-middle-end')
    return f'{build}/pynqo-middle-end'


def build_middle_end(llvm: str = '/usr/lib/llvm-22') -> str:
    """Build pynqo-middle-end with CMake (incrementally); return its Linux path.

    The first build downloads torch-mlir and compiles it, about 13 minutes on the
    Linux file system of a 4 GB WSL. One job at a time is what that was measured
    with; a machine with more memory can raise PYNQO_MIDDLE_END_JOBS.
    """
    import os
    tool = middle_end_tool()
    build = tool.rsplit('/', 1)[0]
    jobs = os.environ.get('PYNQO_MIDDLE_END_JOBS', '1')
    _linux(f"cmake -S '{_linux_path(MIDDLE_END)}' -B {build} -G Ninja -DCMAKE_BUILD_TYPE=Release "
           f"-DCMAKE_C_COMPILER=clang-22 -DCMAKE_CXX_COMPILER=clang++-22 "
           f"-DMLIR_DIR={llvm}/lib/cmake/mlir >/dev/null && cmake --build {build} --target pynqo-middle-end -j {jobs}")
    return tool


def middle_end(torch_mlir: Path, out: Path, tool: str | None = None, emit: str = 'linalg',
               devices: tuple[str, ...] | list[str] = ('cpu',)) -> Path:
    """Run pynqo-middle-end on ``torch_mlir``; write ``out`` and return it.

    ``devices`` is the config's device list. The middle end assigns operations
    to them, cuts each device's subgraphs out into ``delegates_dir(out)/<device>.mlir``
    and leaves delegate calls in ``out``; pynqo-backend refuses a model.mlir
    assigned for another list.
    """
    _linux(f"{tool or middle_end_tool()} '{_linux_path(torch_mlir)}' -o '{_linux_path(out)}' --emit={emit} "
           f"--devices {','.join(devices)} --delegates-dir '{_linux_path(delegates_dir(out))}'")
    return out


def delegates_dir(model_mlir: Path) -> Path:
    """Where the middle end writes, and the back end reads, the device modules of ``model_mlir``."""
    return Path(model_mlir).parent / 'delegates'


def check_scheme(config: dict) -> None:
    """Refuse a package for the NPU in a scheme it cannot run, before any work is done."""
    from src.vendors.npu_pynq.spec import quant as npu_quant
    from src.vendors.risc_v_npu.spec import quant as risc_v_npu_quant

    scheme = config.get('quantization')
    for device, quant in (('npu', npu_quant), ('risc_v_npu', risc_v_npu_quant)):
        if device in config['devices'] and scheme not in quant.ACCEPTED_SCHEMES:
            raise ValueError(f'{device} accepts {quant.ACCEPTED_SCHEMES}, not {scheme!r}')


def calibration_inputs(config: dict, programs: dict, source: str, cache=None) -> dict[str, list[tuple]]:
    """The config's ``calibration:`` block -> {program name: [argument tuples]} for ``quantize``.

    classifier   ``images``: a directory; the first ``count`` of its *.jpg in name
                 order, preprocessed exactly as the runtime feeds an image.
    llm          ``text``: a text file, tokenized with the checkpoint's tokenizer and
                 cut into ``count`` prefill chunks of ``chunk`` tokens. Each chunk
                 continues the KV cache of the one before, as the runtime runs a
                 prompt, and a new sequence starts when the cache is full. Only the
                 prefill program is calibrated; decode shares its layers' scales.
    """
    import numpy as np
    import torch

    spec, graph = config['calibration'], config['graph']
    path = Path(spec.get('images') or spec['text'])
    path = path if path.is_absolute() else REPO / path
    count = int(spec['count'])
    if graph['kind'] == 'classifier':
        from PIL import Image
        from src.runtime.tasks.classifier.preprocess import preprocess

        files = sorted(path.glob('*.jpg'))[:count]
        if not files:
            raise ValueError(f'no calibration images (*.jpg) in {path}')
        return {'forward': [(torch.from_numpy(preprocess(Image.open(f), graph['input'])),) for f in files]}
    from src.runtime.tasks.llm.tokenizer import Tokenizer

    chunk, max_seq = graph['chunk'], graph['max_seq']
    ids = Tokenizer(Path(source) / 'tokenizer.json').encode(path.read_text(encoding='utf-8'))
    if len(ids) < count * chunk:
        raise ValueError(f'{path} has {len(ids)} tokens; {count} chunks of {chunk} need {count * chunk}')
    run = programs['prefill'].module()
    inputs, kv, pos = [], cache.zeros(), 0
    with torch.no_grad():
        for i in range(count):
            if pos + chunk > max_seq:
                kv, pos = cache.zeros(), 0
            tokens = torch.from_numpy(np.asarray([ids[i * chunk:(i + 1) * chunk]], np.int64))
            args = (tokens, torch.tensor([pos]), torch.tensor([chunk - 1]), *kv)
            inputs.append(args)
            kv, pos = list(run(*args)[1:]), pos + chunk
    return {'prefill': inputs}


def export_package(config: dict, out: str | Path, source: str | None = None,
                   llvm_bin: str = '/usr/lib/llvm-22/bin', timings: dict | None = None,
                   profile_kernels: bool = False) -> Path:
    """Run the whole export for ``config`` (examples/<demo>/export.yml); write the package to ``out``.

    load -> export_graph -> quantize -> to_mlir -> pynqo-middle-end -> pynqo-backend -> manifest

    ``profile_kernels`` builds a model.so that times every CPU kernel
    (pynqo-backend --profile-kernels); use it to profile, not to deploy.
    """
    # Imported here so that importing this module does not load torch or torch-mlir.
    import shutil
    from src.export.frontend.export_graph import export_classifier, export_llm
    from src.export.frontend.load import framework, load as load_model
    from src.export.frontend.quantization import quantize
    from src.export.frontend.to_mlir import to_mlir
    from src.vendors.npu_pynq.spec import isa

    check_scheme(config)
    times = timings if timings is not None else {}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # load: a framework builds the float model.
    spec = config['load']
    src = source or spec['source']
    # A relative path that exists is resolved against the repository root.
    # Anything else is kept as written, such as a Hub id or a torchvision weights name.
    if not Path(src).is_absolute() and (REPO / src).exists():
        src = str(REPO / src)
    start = time.monotonic()
    # Any other key under load: is passed to the framework adapter,
    # for example model: for torchvision.
    extra = {k: v for k, v in spec.items() if k not in ('framework', 'source')}
    model = load_model(spec['framework'], src, **extra)
    times['load'] = time.monotonic() - start

    # export_graph: torch.export captures one ExportedProgram per entry point, undecomposed.
    graph = config['graph']
    start = time.monotonic()
    if graph['kind'] == 'llm':
        programs, cache = export_llm(model, graph['max_seq'], graph['chunk'])
    elif graph['kind'] == 'classifier':
        programs, cache = export_classifier(model, tuple(graph['input']['shape'])), None
    else:
        raise ValueError(f"unsupported graph kind {graph['kind']!r}")
    times['export_graph'] = time.monotonic() - start

    # quantize: calibrate on the config's inputs, then a graph pass rewrites each
    # program's linears, convolutions and embeddings.
    scheme = config.get('quantization')
    start = time.monotonic()
    if scheme:
        calibration = calibration_inputs(config, programs, src, cache) if 'calibration' in config else None
        programs = quantize(programs, scheme, calibration)
    times['quantize'] = time.monotonic() - start

    # to_mlir: ATen decompositions, then torch-mlir's importer writes the programs as
    # torch.mlir (the torch dialect).
    start = time.monotonic()
    torch_mlir = to_mlir(programs, out / 'ir')
    times['to_mlir'] = time.monotonic() - start

    # pynqo-middle-end: our rewrites, torch-mlir's pipelines, then device assignment and
    # partitioning for the config's devices: model.mlir (the CPU graph, linalg on tensors,
    # with a delegate call per device subgraph) and ir/delegates/<device>.mlir (the subgraphs).
    start = time.monotonic()
    tool = build_middle_end()
    times['build_middle_end'] = time.monotonic() - start
    start = time.monotonic()
    model_mlir = middle_end(torch_mlir, out / 'ir' / 'model.mlir', tool, devices=config['devices'])
    times['middle_end'] = time.monotonic() - start

    # pynqo-backend: each delegate compiles its device's module, then weights.bin and CPU code.
    # It gets the same device list as the middle end and checks model.mlir was assigned for it.
    start = time.monotonic()
    tool = build_backend()
    times['build_backend'] = time.monotonic() - start
    # The NPU programs are encoded for this array geometry.
    # The runtime checks the overlay against it before loading.
    # These values are always passed, so pynqo-backend's own defaults are not used here.
    limits = isa.Limits()
    start = time.monotonic()
    _linux(f"{tool} '{_linux_path(model_mlir)}' --out '{_linux_path(out)}' "
           f"--delegates-dir '{_linux_path(delegates_dir(model_mlir))}' "
           f"--devices {','.join(config['devices'])} --cpus {','.join(config['cpus'])} "
           f"--llvm-bin {llvm_bin} --rows {limits.rows} --columns {limits.columns} --max-k {limits.max_k}"
           + (" --profile-kernels" if profile_kernels else ""))
    times['backend'] = time.monotonic() - start

    # manifest: add what only the front end knows (runtime files, argument names, model metadata).
    compiled = json.loads((out / 'compile.json').read_text())
    for name, path in framework(spec['framework']).files(src).items():
        shutil.copy2(path, out / name)
    # This is a shallow merge, so manifest and compiled share the entry_points dict.
    # The names added to compiled below therefore reach manifest.json.
    # compile.json on disk keeps what pynqo-backend wrote.
    manifest = {**compiled, 'format': PACKAGE_FORMAT, 'model': config['model'],
                'quantization': scheme, 'compile_seconds': times}
    if graph['kind'] == 'llm':
        # Each list must match the argument order of the signatures in export_graph/export.py.
        # The ABI's leading ws argument is not a named tensor, so it is not listed.
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
