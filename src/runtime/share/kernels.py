"""Kernel timing for packages exported with --profile-kernels.

Such a model.so calls ``__prof_enter(id)`` / ``__prof_exit(id)`` around every
CPU kernel; kernels.json beside it describes each id (op, shapes, the
operations of its body). ``KernelProbes`` loads kernel_probes.c, which defines
those symbols, globally and before the model, as the delegate dispatcher is.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any

from src.runtime.delegate import build_library

HERE = Path(__file__).resolve().parent
TABLE_FORMAT = 'pynqo-kernels/1'


def read_table(path: str | Path) -> list[dict[str, Any]]:
    """kernels.json's entries, where entry ``i`` has id ``i``."""
    data = json.loads(Path(path).read_text())
    if data.get('format') != TABLE_FORMAT:
        raise ValueError(f'{path}: not a {TABLE_FORMAT} kernel table')
    kernels = data['kernels']
    if [k['id'] for k in kernels] != list(range(len(kernels))):
        raise ValueError(f'{path}: kernel ids must be 0..{len(kernels) - 1} in order')
    return kernels


def describe(kernel: dict[str, Any]) -> str:
    """A short label: the op, the operations of its body, its first output type."""
    if kernel['kind'] == 'delegate':
        return f"delegate {kernel.get('delegate', '?')}"
    label = kernel['op']
    body = [op.split('.', 1)[-1] for op in kernel.get('body', [])
            if not op.startswith(('arith.constant', 'linalg.index'))]
    if body:
        label += '(' + ','.join(dict.fromkeys(body)) + ')'
    shape = (kernel.get('outputs') or kernel.get('inputs') or [''])[0]
    return f'{label} {shape}'.strip()


class KernelProbes:
    """The counters behind __prof_enter / __prof_exit, for one kernel table."""

    def __init__(self, table: list[dict[str, Any]]) -> None:
        self.table = table
        path = build_library('kernel_probes', [HERE / 'kernel_probes.c'])
        self.lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self.lib.kp_reset.argtypes = [ctypes.c_int32, ctypes.c_int64]
        self.lib.kp_enable.argtypes = [ctypes.c_int]
        self.lib.kp_stats.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int64)]
        self.lib.kp_events.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
                                       ctypes.POINTER(ctypes.c_int64)]
        self.lib.kp_events.restype = ctypes.c_int64
        self.profile(False)

    def profile(self, enable: bool, events: int = 0) -> None:
        """Start (and clear) or stop timing; ``events`` > 0 also keeps a timeline."""
        if enable:
            if self.lib.kp_reset(len(self.table), int(events)):
                raise MemoryError('cannot allocate the kernel counters')
            self.lib.kp_enable(1)
        else:
            self.lib.kp_enable(0)

    def stats(self) -> list[dict[str, Any]]:
        """Per kernel that ran: its table entry plus calls, total_ns, min_ns, max_ns."""
        out = (ctypes.c_int64 * 4)()
        rows = []
        for kernel in self.table:
            if self.lib.kp_stats(kernel['id'], out) == 0 and out[0]:
                rows.append({**kernel, 'calls': out[0], 'total_ns': out[1], 'min_ns': out[2],
                             'max_ns': out[3]})
        return rows

    def events(self) -> tuple[list[tuple[int, int, int]], int]:
        """Recorded (id, begin_ns, end_ns) events and how many were dropped."""
        dropped = ctypes.c_int64(0)
        count = self.lib.kp_events(None, 0, ctypes.byref(dropped))
        out = (ctypes.c_int64 * (3 * max(count, 1)))()
        self.lib.kp_events(out, count, None)
        return [(out[3 * i], out[3 * i + 1], out[3 * i + 2]) for i in range(count)], int(dropped.value)
