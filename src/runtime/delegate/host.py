"""The host delegate backend: every delegate runs as an INT8 GEMM on this CPU.

Each delegated subgraph is one ``M x K`` INT8 by ``K x N`` INT8 matmul with an
``M x N`` INT32 result (the ``pynqo-package/3`` delegate contract). The device's
program table gives each program's shape and where its B lives in
``weights.bin``; B is stored as ``spec/layout.py``'s ``pack_b`` (column-tile
panels, the same for npu and risc_v_npu), so it is unpacked once and the call
is a plain numpy matmul. Nothing of the device's own program (ISA words or
RISC-V binaries) is run, so a package for any delegate device runs anywhere.

The product wraps modulo 2^32, as the NPU's C planes summed by libnpu_rt and
librv_npu_rt do.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from src.vendors.npu_pynq.spec.layout import GemmPlan

from . import DELEGATE_FN, DelegateError

#: Device -> (program table, its header (tag, format) or None when not
#: checked, index of weight_offset in its ``seg`` lines); b_bytes follows it.
#:   npu:        seg <word_offset> <length> <weight_offset> <b_bytes> <nrel> <rel>...
#:   risc_v_npu: seg <ni_lo> <ni_hi> <weight_offset> <b_bytes>   (rv-programs format 2)
TABLES = {'npu': ('programs.txt', None, 3), 'risc_v_npu': ('rv_programs.txt', ('rv-programs', '2'), 3)}


def read_programs(package: Path, device: str) -> list[tuple[GemmPlan, list[tuple[int, int]]]]:
    """Per program index, its GEMM plan and its segments' (weight_offset, b_bytes)."""
    if device not in TABLES:
        raise DelegateError(f'the host backend does not know device {device!r}')
    name, header, field = TABLES[device]
    path = Path(package) / name
    try:
        lines = path.read_text().split('\n')
    except OSError as error:
        raise DelegateError(f'cannot read {path}: {error}') from error
    if header is not None and lines[0].split()[:2] != list(header):
        raise DelegateError(f'{name}: expected format {" ".join(header)}, found {lines[0][:40]!r}; '
                            'export the model again')
    programs = []
    for line in lines[1:]:
        words = line.split()
        if not words:
            continue
        if words[0] == 'task':
            if int(words[1]) != len(programs):
                raise DelegateError(f'{name}: tasks out of order')
            m, n, k, tm, tn, tk = (int(w) for w in words[2:8])
            programs.append((GemmPlan(m, n, k, tm, tn, tk), []))
        elif words[0] == 'seg' and programs:
            programs[-1][1].append((int(words[field]), int(words[field + 1])))
        else:
            raise DelegateError(f'{name}: unexpected line {line!r}')
    return programs


def unpack_b(plan: GemmPlan, packed: np.ndarray) -> np.ndarray:
    """The ``K x N`` INT8 matrix that ``pack_b`` laid out as ``packed``."""
    b = np.empty((plan.k, plan.n), np.int8)
    for kc in range(plan.kt):
        r0, rows = kc * plan.tk, plan.k_c(kc)
        for ni in range(plan.nt):
            c0, cols = ni * plan.tn, plan.n_i(ni)
            off = plan.b_offset(kc, ni)
            b[r0:r0 + rows, c0:c0 + cols] = packed[off:off + rows * cols].reshape(rows, cols)
    return b


class HostBackend:
    """Runs a package's delegates for ``device`` on this CPU."""

    clock_hz = None   # no device clock: cycles stay 0

    def __init__(self, package: Path, device: str) -> None:
        self.device = device
        self.programs = read_programs(package, device)
        self.weights = np.memmap(Path(package) / 'weights.bin', dtype=np.int8, mode='r')
        self.b: dict[int, np.ndarray] = {}
        self.calls = self.macs = 0
        self.error = 0
        self.error_program = -1
        self.message = ''
        self._fn = DELEGATE_FN(self._execute)   # keep the callback alive
        self.execute = ctypes.cast(self._fn, ctypes.c_void_p)

    def matrix(self, index: int) -> np.ndarray:
        """Program ``index``'s B as an INT32 ``K x N`` matrix, unpacked on first use."""
        b = self.b.get(index)
        if b is None:
            plan, segments = self.programs[index]
            # Segments cover consecutive column-tile ranges whose panels are
            # contiguous, so their blocks join into the program's whole pack_b.
            packed = np.concatenate([self.weights[o:o + n] for o, n in segments])
            if packed.size != plan.b_bytes:
                raise DelegateError(f'program {index}: B holds {packed.size} bytes, expected {plan.b_bytes}')
            b = self.b[index] = unpack_b(plan, packed).astype(np.int32)
        return b

    def _execute(self, index: int, buffers, count: int) -> int:
        try:
            plan = self.programs[index][0]
            a = np.ctypeslib.as_array(ctypes.cast(buffers[0], ctypes.POINTER(ctypes.c_int8)),
                                      shape=(plan.m, plan.k))
            out = np.ctypeslib.as_array(ctypes.cast(buffers[1], ctypes.POINTER(ctypes.c_int32)),
                                        shape=(plan.m, plan.n))
            np.matmul(a.astype(np.int32), self.matrix(index), out=out)
            self.calls += 1
            self.macs += plan.m * plan.n * plan.k
            return 0
        except Exception as error:   # an exception cannot cross the C callback
            self.error, self.error_program, self.message = 1, int(index), f'{type(error).__name__}: {error}'
            return 1

    def stats(self) -> dict[str, int]:
        return {'calls': self.calls, 'jobs': 0, 'cycles': 0, 'macs': self.macs, 'polls': 0,
                'error': self.error, 'error_program': self.error_program}

    def close(self) -> None:
        """Drop the unpacked matrices and unmap weights.bin."""
        self.b = {}
        self.weights = None
