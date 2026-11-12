"""The risc_v_npu delegate backend: librv_npu_rt (src/vendors/risc_v_npu/driver/rv_npu_rt).

The device is the NPU built with the CVA6 RISC-V scalar core: each delegate
call runs RISC-V binaries on CVA6, which issue the GEMM jobs. As for npu, the
overlay is loaded with ``pynq.Overlay`` and the buffers CVA6 and the NPU reach
are allocated with ``pynq.allocate``; they are handed to ``rv_npu_rt_load``
through its C interface. There is no simulator: npu_pynq's SIM is a model of
the 64-bit ISA and cannot run RISC-V, so a package for risc_v_npu runs only on
the board with the CVA6 overlay.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any

from src.vendors.risc_v_npu.spec.abi import Capability, REQUIRED_CAPABILITIES, Register
from src.vendors.risc_v_npu.spec.board import PL_CLOCK_HZ

from . import DelegateError, build_library

DRIVER = Path(__file__).resolve().parents[2] / 'vendors' / 'risc_v_npu' / 'driver' / 'rv_npu_rt'
SOURCES = ('load.cpp', 'execute.cpp', 'pynq.cpp', 'profile.cpp')
PHASE_FIELDS = ('m', 'n', 'k', 'segments', 'calls', 'call_ns_min', 'call_ns_max',
                'pack_a_ns', 'stage_b_ns', 'pl_ns', 'unpack_c_ns', 'pl_cycles', 'polls', 'jobs',
                'pack_a_bytes', 'stage_b_bytes', 'unpack_c_bytes')   # rv_npu_rt.h, RV_NPU_RT_PHASE_FIELDS
PHASES = ('pack_a', 'stage_b', 'pl', 'unpack_c')                     # rv_npu_rt_events' phase ids
NO_SIM = ("risc_v_npu has no SIM backend: npu_pynq's SIM is an ISA model and cannot run RISC-V; "
          "run the package on the board with the CVA6 overlay, or export it for [cpu, npu]")


class Buffer(ctypes.Structure):
    _fields_ = [('virt', ctypes.c_void_p), ('phys', ctypes.c_uint32), ('size', ctypes.c_uint32)]


class Platform(ctypes.Structure):
    _fields_ = [('mmio', ctypes.c_void_p), ('io', Buffer), ('code', Buffer)]


def check_overlay(overlay: Any, accelerator: str, package: Path) -> None:
    """The overlay has the CVA6 scalar core and the array the package was compiled for."""
    capabilities = int(getattr(overlay, accelerator).read(Register.CAPABILITIES))
    if not capabilities & Capability.RISCV_SCALAR_CORE:
        raise DelegateError('the overlay has no CVA6 scalar core (capability bit 7, '
                            'RISCV_SCALAR_CORE); load the CVA6 build or export for [cpu, npu]')
    missing = Capability(REQUIRED_CAPABILITIES) & ~Capability(capabilities)
    if missing:
        raise DelegateError(f'the overlay lacks required capabilities: {missing!s}')
    params = overlay.ip_dict[accelerator].get('parameters', {})
    found = {'rows': int(params.get('ROWS', 16)), 'columns': int(params.get('COLUMNS', 16)),
             'max_k': int(params.get('MAX_K', 256))}
    wanted = json.loads((Path(package) / 'manifest.json').read_text()).get('npu_limits', found)
    if found != wanted:
        raise DelegateError(f'the package was compiled for {wanted}, the overlay has {found}')


class RiscvNpuBackend:
    clock_hz = PL_CLOCK_HZ   # what ISA_CYCLES counts

    def __init__(self, package: Path, overlay: Any = None, accelerator: str = 'npu_accelerator_0') -> None:
        if overlay is None:
            raise DelegateError(NO_SIM)
        check_overlay(overlay, accelerator, package)
        sources = [DRIVER / name for name in SOURCES] + [DRIVER / 'rv_npu_rt.h', DRIVER / 'CMakeLists.txt']
        self.lib = ctypes.CDLL(str(build_library('rv_npu_rt', sources, cxx=True)))
        lib = self.lib
        lib.rv_npu_rt_requirements.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)]
        lib.rv_npu_rt_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(Platform)]
        lib.rv_npu_rt_error.restype = ctypes.c_char_p
        lib.rv_npu_rt_set_timeout_ms.argtypes = [ctypes.c_int64]
        lib.rv_npu_rt_stats.argtypes = [ctypes.POINTER(ctypes.c_int64)]
        lib.rv_npu_rt_profile.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.rv_npu_rt_program_count.restype = ctypes.c_int64
        lib.rv_npu_rt_phase_stats.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]
        lib.rv_npu_rt_events.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
                                         ctypes.POINTER(ctypes.c_int64)]
        lib.rv_npu_rt_events.restype = ctypes.c_int64
        root = str(Path(package).resolve()).encode()
        sizes = (ctypes.c_uint32 * 2)()
        if lib.rv_npu_rt_requirements(root, sizes):
            raise DelegateError(lib.rv_npu_rt_error().decode())
        from pynq import allocate

        self.buffers = []   # pynq buffers must outlive the library's use of them
        platform = Platform()
        platform.mmio = getattr(overlay, accelerator).mmio.array.ctypes.data
        for name, size in (('io', sizes[0]), ('code', sizes[1])):
            buf = allocate(shape=(max(int(size), 8),), dtype='u1', cacheable=False)
            self.buffers.append(buf)
            setattr(platform, name, Buffer(buf.ctypes.data, int(buf.physical_address), int(size)))
        self.platform = platform
        self.mmio = int(platform.mmio)
        if lib.rv_npu_rt_load(root, ctypes.byref(platform)):
            message = lib.rv_npu_rt_error().decode()
            self.close()
            raise DelegateError('rv_npu_rt_load: ' + message)
        self.execute = ctypes.cast(lib.rv_npu_rt_execute, ctypes.c_void_p)

    def set_timeout(self, milliseconds: int) -> None:
        """Give up on a RISC-V program after this long (0 waits forever)."""
        self.lib.rv_npu_rt_set_timeout_ms(int(milliseconds))

    def stats(self) -> dict[str, int]:
        out = (ctypes.c_int64 * 7)()
        self.lib.rv_npu_rt_stats(out)
        return dict(zip(('calls', 'jobs', 'cycles', 'macs', 'polls', 'error', 'error_program'), out))

    def profile(self, enable: bool, events: int = 0) -> None:
        """Start (and clear) or stop phase profiling; ``events`` > 0 also keeps a timeline."""
        self.lib.rv_npu_rt_profile(int(enable), int(events))

    def phase_stats(self) -> list[dict[str, int]]:
        """Per program index, its shape and the phase sums since profiling started."""
        out = (ctypes.c_int64 * len(PHASE_FIELDS))()
        rows = []
        for index in range(self.lib.rv_npu_rt_program_count()):
            if self.lib.rv_npu_rt_phase_stats(index, out) == 0:
                rows.append({'index': index, **dict(zip(PHASE_FIELDS, out))})
        return rows

    def events(self) -> tuple[list[tuple[int, str, int, int]], int]:
        """Recorded (index, phase, begin_ns, end_ns) events and how many were dropped."""
        dropped = ctypes.c_int64(0)
        count = self.lib.rv_npu_rt_events(None, 0, ctypes.byref(dropped))
        out = (ctypes.c_int64 * (4 * max(count, 1)))()
        self.lib.rv_npu_rt_events(out, count, None)
        return [(out[4 * i], PHASES[out[4 * i + 1]], out[4 * i + 2], out[4 * i + 3])
                for i in range(count)], int(dropped.value)

    def close(self) -> None:
        self.lib.rv_npu_rt_close()
        for buf in self.buffers:
            buf.freebuffer()
        self.buffers = []
