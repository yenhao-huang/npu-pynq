"""The npu delegate backend: libnpu_rt (src/vendors/npu_pynq/driver/npu_rt).

On the board the overlay is loaded with ``pynq.Overlay`` and the buffers the
NPU reaches are allocated with ``pynq.allocate`` (PYNQ's standard way); they
are handed to ``npu_rt_load`` through its C interface. Elsewhere libnpu_rt
runs its ISA simulator on host memory.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from src.vendors.npu_pynq.spec.abi import Capability, Register

from . import DelegateError, build_library

DRIVER = Path(__file__).resolve().parents[2] / 'vendors' / 'npu_pynq' / 'driver' / 'npu_rt'
SOURCES = ('load.cpp', 'execute.cpp', 'pynq.cpp', 'sim.cpp')


class Buffer(ctypes.Structure):
    _fields_ = [('virt', ctypes.c_void_p), ('phys', ctypes.c_uint32), ('size', ctypes.c_uint32)]


class Platform(ctypes.Structure):
    _fields_ = [('mmio', ctypes.c_void_p), ('io', Buffer), ('stage', Buffer), ('code', Buffer)]




def check_overlay(overlay: Any, accelerator: str, package: Path) -> None:
    """The overlay runs ISA programs and has the array the package was compiled for."""
    import json

    if not int(getattr(overlay, accelerator).read(Register.CAPABILITIES)) & Capability.ISA_FRONTEND:
        raise DelegateError('the overlay does not implement the ISA front end (ABI 1.1)')
    params = overlay.ip_dict[accelerator].get('parameters', {})
    found = {'rows': int(params.get('ROWS', 16)), 'columns': int(params.get('COLUMNS', 16)),
             'max_k': int(params.get('MAX_K', 256))}
    wanted = json.loads((Path(package) / 'manifest.json').read_text()).get('npu_limits', found)
    if found != wanted:
        raise DelegateError(f'the package was compiled for {wanted}, the overlay has {found}')


class NpuBackend:
    def __init__(self, package: Path, overlay: Any = None, accelerator: str = 'npu_accelerator_0') -> None:
        sources = [DRIVER / name for name in SOURCES] + [DRIVER / 'npu_rt.h', DRIVER / 'CMakeLists.txt']
        self.lib = ctypes.CDLL(str(build_library('npu_rt', sources, cxx=True)))
        lib = self.lib
        lib.npu_rt_requirements.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)]
        lib.npu_rt_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(Platform)]
        lib.npu_rt_error.restype = ctypes.c_char_p
        lib.npu_rt_stats.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        root = str(Path(package).resolve()).encode()
        self.buffers = []   # pynq buffers must outlive the library's use of them
        platform = None
        if overlay is not None:
            check_overlay(overlay, accelerator, package)
            sizes = (ctypes.c_uint32 * 3)()
            if lib.npu_rt_requirements(root, sizes):
                raise DelegateError(lib.npu_rt_error().decode())
            platform = Platform()
            platform.mmio = getattr(overlay, accelerator).mmio.array.ctypes.data
            for name, size, cacheable in (('io', sizes[0], False), ('stage', sizes[1], False),
                                          ('code', sizes[2], False)):
                from pynq import allocate
                buf = allocate(shape=(max(int(size), 8),), dtype='u1', cacheable=cacheable)
                self.buffers.append(buf)
                setattr(platform, name, Buffer(buf.ctypes.data, int(buf.physical_address), int(size)))
        self.platform = platform   # None: the simulator
        self.mmio = int(platform.mmio) if platform is not None else 0
        if lib.npu_rt_load(root, ctypes.byref(platform) if platform is not None else None):
            raise DelegateError('npu_rt_load: ' + lib.npu_rt_error().decode())
        self.execute = ctypes.cast(lib.npu_rt_execute, ctypes.c_void_p)

    def stats(self) -> dict[str, int]:
        out = (ctypes.c_int64 * 7)()
        self.lib.npu_rt_stats(out)
        return dict(zip(('calls', 'jobs', 'cycles', 'macs', 'polls', 'error', 'error_program'), out))

    def close(self) -> None:
        self.lib.npu_rt_close()
        for buf in self.buffers:
            buf.freebuffer()
        self.buffers = []
