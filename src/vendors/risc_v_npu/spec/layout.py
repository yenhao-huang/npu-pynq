"""Tiled GEMM layout contract shared by the encoder, runtimes and tests.

A logical ``C[M, N] = A[M, K] @ B[K, N]`` (INT8 x INT8 -> INT32) is split into
``Mt x Kt x Nt`` jobs. Tile ``(mi, kc)`` of A, ``(kc, ni)`` of B and
``(kc, mi, ni)`` of C are dense row-major blocks:

``A``  ``a_base + a_offset(mi, kc)``, ``m_i x k_c`` bytes
``B``  ``b_base + b_offset(kc, ni)``, ``k_c x n_i`` bytes
``C``  ``c_base + c_offset(kc, mi, ni)``, ``m_i x n_i`` INT32 partial sums

Every block starts 8-byte aligned. B is stored column-tile-major: column tile
``ni`` is one contiguous panel holding its K chunks in order, so any range of
output columns is one contiguous block of the weight (a *segment* that can be
streamed into a small DMA window), and a REPEATed GEMM with post-increment
walks the tiles of one K chunk by stepping one panel. C keeps one partial
plane per K chunk; librv_npu_rt sums the planes while unpacking C, so the
delegate's output is the ``M x N`` product (``unpack_c`` below returns the
planes apart, for tests).

Copied from src/vendors/npu_pynq/spec/layout.py (risc_v_npu keeps its own
copy; plan 2026-1112-003 section 6.2). Changes from the source: ``align``,
``Limits`` and ``LayoutError`` are defined here instead of coming from
npu_pynq's ``isa.py`` (this device has no 64-bit ISA), the ISA encoder
``encode`` is left out (the RISC-V program replaces it), and ``DataLayout``
states where A, C and the staged B segment sit relative to ``data_base``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Every A, B and C block starts on this boundary (the DMA's beat).
ALIGN = 8


class LayoutError(ValueError):
    """A GEMM does not fit the matrix core's limits."""


@dataclass(frozen=True)
class Limits:
    """Physical limits of one implementation, read from the overlay."""

    rows: int = 16
    columns: int = 16
    max_k: int = 256


def align(value: int, to: int = ALIGN) -> int:
    return (value + to - 1) // to * to


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class GemmPlan:
    m: int
    n: int
    k: int
    tm: int
    tn: int
    tk: int

    @property
    def mt(self) -> int:
        return _ceil_div(self.m, self.tm)

    @property
    def nt(self) -> int:
        return _ceil_div(self.n, self.tn)

    @property
    def kt(self) -> int:
        return _ceil_div(self.k, self.tk)

    def m_i(self, mi: int) -> int:
        return min(self.tm, self.m - mi * self.tm)

    def n_i(self, ni: int) -> int:
        return min(self.tn, self.n - ni * self.tn)

    def k_c(self, kc: int) -> int:
        return min(self.tk, self.k - kc * self.tk)

    # -- A ----------------------------------------------------------------
    def a_tile_stride(self, mi: int) -> int:
        return align(self.m_i(mi) * self.tk)

    def a_offset(self, mi: int, kc: int) -> int:
        # Full tile rows precede row mi; within a row, chunks share a stride.
        return mi * self.kt * align(self.tm * self.tk) + kc * self.a_tile_stride(mi)

    @property
    def a_bytes(self) -> int:
        last = self.mt - 1
        return self.a_offset(last, self.kt - 1) + align(self.m_i(last) * self.k_c(self.kt - 1))

    # -- B ----------------------------------------------------------------
    def b_tile_stride(self, kc: int) -> int:
        return align(self.k_c(kc) * self.tn)

    @property
    def b_panel(self) -> int:
        """Bytes of one column tile's panel (all K chunks)."""
        return (self.kt - 1) * align(self.tk * self.tn) + self.b_tile_stride(self.kt - 1)

    def b_offset(self, kc: int, ni: int) -> int:
        return ni * self.b_panel + kc * align(self.tk * self.tn)

    @property
    def b_bytes(self) -> int:
        return self.nt * self.b_panel

    def segments(self, limit: int) -> list[tuple[int, int]]:
        """Column-tile ranges whose B blocks are at most ``limit`` bytes (>= 1 tile)."""
        per = max(1, limit // self.b_panel)
        return [(lo, min(lo + per, self.nt)) for lo in range(0, self.nt, per)]

    # -- C ----------------------------------------------------------------
    def c_tile_stride(self, mi: int) -> int:
        return align(4 * self.m_i(mi) * self.tn)

    def c_plane_bytes(self) -> int:
        return sum(self.nt * self.c_tile_stride(mi) for mi in range(self.mt))

    def c_offset(self, kc: int, mi: int, ni: int) -> int:
        row = sum(self.nt * self.c_tile_stride(r) for r in range(mi))
        return kc * self.c_plane_bytes() + row + ni * self.c_tile_stride(mi)

    @property
    def c_bytes(self) -> int:
        return self.kt * self.c_plane_bytes()

    @property
    def jobs(self) -> int:
        return self.mt * self.nt * self.kt


def plan_gemm(m: int, n: int, k: int, limits: Limits = Limits(), tk: int | None = None) -> GemmPlan:
    if min(m, n, k) < 1:
        raise LayoutError("GEMM dimensions must be positive")
    tk = limits.max_k if tk is None else tk
    if not 1 <= tk <= limits.max_k:
        raise LayoutError(f"K chunk {tk} exceeds MAX_K {limits.max_k}")
    return GemmPlan(m, n, k, min(m, limits.rows), min(n, limits.columns), min(k, tk))


def pack_a(plan: GemmPlan, a: np.ndarray) -> np.ndarray:
    out = np.zeros(plan.a_bytes, np.int8)
    for mi in range(plan.mt):
        r0, rows = mi * plan.tm, plan.m_i(mi)
        for kc in range(plan.kt):
            c0, cols = kc * plan.tk, plan.k_c(kc)
            off = plan.a_offset(mi, kc)
            out[off:off + rows * cols] = a[r0:r0 + rows, c0:c0 + cols].reshape(-1)
    return out


def pack_b(plan: GemmPlan, b: np.ndarray) -> np.ndarray:
    out = np.zeros(plan.b_bytes, np.int8)
    for kc in range(plan.kt):
        r0, rows = kc * plan.tk, plan.k_c(kc)
        for ni in range(plan.nt):
            c0, cols = ni * plan.tn, plan.n_i(ni)
            off = plan.b_offset(kc, ni)
            out[off:off + rows * cols] = b[r0:r0 + rows, c0:c0 + cols].reshape(-1)
    return out


def unpack_c(plan: GemmPlan, buffer: np.ndarray) -> np.ndarray:
    """Return the ``[Kt, M, N]`` INT32 partial-sum planes."""
    words = np.asarray(buffer).view(np.uint8)[: plan.c_bytes]
    out = np.zeros((plan.kt, plan.m, plan.n), np.int32)
    for kc in range(plan.kt):
        for mi in range(plan.mt):
            r0, rows = mi * plan.tm, plan.m_i(mi)
            for ni in range(plan.nt):
                c0, cols = ni * plan.tn, plan.n_i(ni)
                off = plan.c_offset(kc, mi, ni)
                tile = words[off:off + 4 * rows * cols].view("<i4").reshape(rows, cols)
                out[kc, r0:r0 + rows, c0:c0 + cols] = tile
    return out


#: Alignment of the C planes and of the B stage window inside the data block.
BLOCK_ALIGN = 64


@dataclass(frozen=True)
class DataLayout:
    """One delegate's data block, as offsets from ``data_base``.

    Unlike npu_pynq, whose loader patches every ADDR_B word to wherever the
    stage buffer sits, a RISC-V program is not patched after it is linked: it
    reads ``DATA_BASE`` and adds these constant offsets. So the staged B
    segment lives inside the same block, after C. The driver copies segment
    ``s``'s weights to ``data_base + b_base`` before running segment ``s``.
    """

    a_base: int
    a_bytes: int
    c_base: int
    c_bytes: int
    b_base: int
    b_bytes: int      # the largest segment's B block

    @property
    def total_bytes(self) -> int:
        return self.b_base + self.b_bytes


def data_layout(plan: GemmPlan, segment_limit: int) -> DataLayout:
    """A at 0, C at the next 64-byte boundary, the B stage window after C."""
    c_base = align(plan.a_bytes, BLOCK_ALIGN)
    b_base = align(c_base + plan.c_bytes, BLOCK_ALIGN)
    b_bytes = max((hi - lo) * plan.b_panel for lo, hi in plan.segments(segment_limit))
    return DataLayout(0, plan.a_bytes, c_base, plan.c_bytes, b_base, b_bytes)
