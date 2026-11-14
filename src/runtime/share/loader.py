"""Load and call a package exported through PyTorch (format pynqo-package/3).

The package holds model.so per CPU model, weights.bin and, for each delegate
device, its own artefacts (programs.* for npu, rv_programs.* for risc_v_npu).
Entry points take memref descriptors through MLIR's C interface:

    _mlir_ciface_<entry>(ws, *inputs, *outputs)

This module knows no task: tasks/llm and tasks/classifier build their engines on
ExportedPackage.call and ExportedPackage.stats.
"""

from __future__ import annotations

import ctypes
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from src.runtime.delegate import Dispatcher, backend_class

# 3: a delegate's output is its subgraph's result (libnpu_rt sums the NPU's
# K chunks); /2 packages hand the CPU partial planes and are refused.
FORMAT = 'pynqo-package/3'
DTYPES = {'f32': np.float32, 'i64': np.int64, 'i32': np.int32, 'i8': np.int8}


def descriptor_type(rank: int) -> type:
    fields = [('allocated', ctypes.c_void_p), ('aligned', ctypes.c_void_p), ('offset', ctypes.c_ssize_t)]
    if rank:
        fields += [('sizes', ctypes.c_ssize_t * rank), ('strides', ctypes.c_ssize_t * rank)]
    return type(f'MemRef{rank}D', (ctypes.Structure,), {'_fields_': fields})


def descriptor(array: np.ndarray) -> ctypes.Structure:
    if not array.flags.c_contiguous:
        raise ValueError('memref arguments must be C-contiguous')
    desc = descriptor_type(array.ndim)()
    desc.allocated = desc.aligned = array.ctypes.data
    desc.offset = 0
    for i, (size, stride) in enumerate(zip(array.shape, array.strides)):
        desc.sizes[i] = size
        desc.strides[i] = stride // array.itemsize
    return desc


def cpu_model() -> str:
    machine = platform.machine()
    return 'arm-cortex-a9' if machine.startswith('armv7') else machine


def on_board() -> bool:
    return Path('/etc/profile.d/pynq_venv.sh').exists()


class ExportedPackage:
    """model.so with its weights and delegate backends, ready to call."""

    def __init__(self, root: str | Path, overlay: Any = None) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('format') != FORMAT:
            raise ValueError(f'{self.root}: not an {FORMAT} package')
        self.dispatcher = Dispatcher()
        self.backends = {}
        devices = {d['device'] for d in self.manifest['delegates']}
        if {'npu', 'risc_v_npu'} <= devices:
            raise ValueError(f'{self.root}: npu and risc_v_npu need different overlays; '
                             'a package targets one of them')
        for device in sorted(devices):
            cls = backend_class(device)
            if cls is not None:   # a device without one fails in install() below
                self.backends[device] = cls(self.root, overlay)
        self.dispatcher.install(self.manifest['delegates'],
                                {name: b.execute for name, b in self.backends.items()})
        libraries = self.manifest['libraries']
        model = cpu_model()
        name = next((k for k in libraries if k == model or k.startswith(model + '-')), None)
        if name is None:
            raise ValueError(f'no library for this CPU ({model}); package has {sorted(libraries)}')
        if model == 'arm-cortex-a9':
            for dep in ('libm.so.6', 'libgcc_s.so.1'):
                ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL(str(self.root / libraries[name]))
        weights = self.root / 'weights.bin'
        self.weights = np.memmap(weights, dtype=np.uint8, mode='r') if weights.stat().st_size else np.zeros(8, np.uint8)

    def call(self, entry: str, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
        fn = getattr(self.lib, f'_mlir_ciface_{entry}')
        descs = [descriptor(a) for a in (self.weights, *inputs, *outputs)]
        fn.restype = None
        fn(*(ctypes.byref(d) for d in descs))
        self.dispatcher.check()

    def stats(self) -> dict[str, int]:
        npu = self.backends.get('npu') or self.backends.get('risc_v_npu')
        stats = npu.stats() if npu else {}
        stats['delegate_calls'] = self.dispatcher.calls
        return stats

    def profile(self, enable: bool, events: int = 0) -> None:
        """Start or stop phase profiling in every backend that supports it."""
        for backend in self.backends.values():
            if hasattr(backend, 'profile'):
                backend.profile(enable, events)

    def phase_stats(self) -> dict[str, list[dict[str, int]]]:
        return {name: b.phase_stats() for name, b in self.backends.items() if hasattr(b, 'phase_stats')}

    def events(self) -> dict[str, tuple[list[tuple[int, str, int, int]], int]]:
        return {name: b.events() for name, b in self.backends.items() if hasattr(b, 'events')}

    def close(self) -> None:
        """Release the backends' board buffers; the package cannot be called afterwards."""
        for backend in self.backends.values():
            backend.close()
        self.backends = {}
