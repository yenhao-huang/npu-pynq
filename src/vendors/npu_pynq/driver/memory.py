"""Buffers the NPU can reach: CMA on the board, page-aligned host memory in sim.

Every buffer exposes ``array`` (a uint8 view), ``address`` (its CPU virtual
address), ``physical_address`` (what the NPU sees) and ``flush()``.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


class MappedWeights:
    """weights.bin mapped read-only: pages load on demand and are never copied."""

    def __init__(self, path: Path) -> None:
        self.array = np.memmap(path, dtype=np.uint8, mode="r") if path.stat().st_size else np.zeros(8, np.uint8)
        self.physical_address = 0

    @property
    def address(self) -> int:
        return self.array.ctypes.data

    def flush(self) -> None:
        pass


class HostBuffer:
    """Page-aligned host memory standing in for a CMA buffer."""

    def __init__(self, nbytes: int, phys: int) -> None:
        self._raw = np.zeros(nbytes + 4096, np.uint8)
        start = (-self._raw.ctypes.data) % 4096
        self.array = self._raw[start:start + nbytes]
        self.physical_address = phys

    @property
    def address(self) -> int:
        return self.array.ctypes.data

    def flush(self) -> None:
        pass


class CmaBuffer:
    """A pynq.allocate buffer; uncached ones need no flush/invalidate."""

    def __init__(self, nbytes: int, cacheable: bool) -> None:
        from pynq import allocate
        self.buffer = allocate(shape=(max(nbytes, 8),), dtype=np.uint8, cacheable=cacheable)
        self.array = np.asarray(self.buffer)[:nbytes]
        self.physical_address = int(self.buffer.physical_address)

    @property
    def address(self) -> int:
        return self.array.ctypes.data

    def flush(self) -> None:
        self.buffer.flush()
