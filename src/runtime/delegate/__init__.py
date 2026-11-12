"""Delegates at run time: the __delegate__ dispatcher and the device backends.

model.so calls ``__delegate__(id, in, out)`` for every region a delegate runs.
``Dispatcher`` loads the small C library that defines it (dispatch.c, built
here on first use with the machine's own compiler) and routes each id to the
backend the package's delegate table names. ``BACKENDS`` maps each device to
its backend class: ``npu`` is libnpu_rt (npu.py, the CUSTOM scalar core) and
``risc_v_npu`` is librv_npu_rt (risc_v_npu.py, the CVA6 scalar core).
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DELEGATE_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int64, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int)


#: Device name in the package's delegate table -> (module, class) of its
#: backend, imported only when a package uses that device.
BACKENDS = {
    'npu': ('src.runtime.delegate.npu', 'NpuBackend'),
    'risc_v_npu': ('src.runtime.delegate.risc_v_npu', 'RiscvNpuBackend'),
}


class DelegateError(RuntimeError):
    pass


def backend_class(device: str) -> type | None:
    """The backend class for ``device``, or None for a device with no backend."""
    entry = BACKENDS.get(device)
    if entry is None:
        return None
    module, name = entry
    return getattr(importlib.import_module(module), name)


def build_dir() -> Path:
    """Where native libraries are built: beside the runtime, else the user cache."""
    for root in (HERE.parents[2] / 'build' / 'native', Path.home() / '.cache' / 'npu-native'):
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root
        except OSError:
            continue
    raise DelegateError('no writable directory for native libraries')


def build_library(name: str, sources: list[Path], cxx: bool = False) -> Path:
    """Compile ``sources`` into lib<name>.so unless an up-to-date one exists.

    CMake builds it when the sources carry a CMakeLists.txt and cmake is
    installed; otherwise the system compiler is called directly.
    """
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(path.name.encode() + path.read_bytes())
    out = build_dir() / f'{name}-{digest.hexdigest()[:12]}'
    library = out / f'lib{name}.so'
    if library.exists():
        return library
    out.mkdir(parents=True, exist_ok=True)
    src_dir = sources[0].parent
    if (src_dir / 'CMakeLists.txt').exists() and shutil.which('cmake'):
        work = out / 'cmake'
        for args in (['cmake', '-S', str(src_dir), '-B', str(work), '-DCMAKE_BUILD_TYPE=Release'],
                     ['cmake', '--build', str(work)]):
            proc = subprocess.run(args, capture_output=True, text=True)
            if proc.returncode:
                raise DelegateError(f'building {name} failed:\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}')
        shutil.copy2(work / f'lib{name}.so', library)
        return library
    compiler = ('c++' if cxx else 'cc')
    args = [compiler, '-O2', '-shared', '-fPIC', *(['-std=c++17'] if cxx else []),
            *(str(s) for s in sources if s.suffix in ('.c', '.cpp')), '-o', str(library)]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode:
        raise DelegateError(f'building {name} failed:\n{proc.stderr[-4000:]}')
    return library


class Dispatcher:
    """dispatch.c, loaded globally so model.so resolves __delegate__ against it."""

    def __init__(self) -> None:
        if not sys.platform.startswith('linux'):
            raise DelegateError('delegate dispatch needs Linux (the board or a Linux export host)')
        path = build_library('npu_delegate', [HERE / 'dispatch.c'])
        self.lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self.lib.delegate_reset.argtypes = [ctypes.c_int64]
        self.lib.delegate_register.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64]
        self.lib.delegate_status.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        self.lib.delegate_calls.restype = ctypes.c_int64

    def install(self, table: list[dict], backends: dict[str, ctypes.c_void_p]) -> None:
        if self.lib.delegate_reset(len(table)):
            raise DelegateError('cannot allocate the delegate table')
        for entry in table:
            fn = backends.get(entry['device'])
            if fn is None:
                raise DelegateError(f"no backend for device {entry['device']!r}")
            if self.lib.delegate_register(entry['id'], fn, entry['index']):
                raise DelegateError(f"cannot register delegate {entry['id']}")

    def check(self) -> None:
        failed = ctypes.c_int64(-1)
        status = self.lib.delegate_status(ctypes.byref(failed))
        if status:
            self.lib.delegate_clear()
            raise DelegateError(f'delegate {failed.value} failed with status 0x{status & 0xffffffff:x}')

    @property
    def calls(self) -> int:
        return int(self.lib.delegate_calls())
