"""Tiled GEMM layout contract shared by the encoder, runtimes and tests.

A logical ``C[M, N] = A[M, K] @ B[K, N]`` (INT8 x INT8 -> INT32) is split into
``Mt x Kt x Nt`` jobs. Tile ``(mi, kc)`` of A, ``(kc, ni)`` of B and
``(kc, mi, ni)`` of C are dense row-major blocks:

``A``  ``a_base + a_offset(mi, kc)``, ``m_i x k_c`` bytes
``B``  ``b_base + b_offset(kc, ni)``, ``k_c x n_i`` bytes
``C``  ``c_base + c_offset(kc, mi, ni)``, ``m_i x n_i`` INT32 partial sums

Every block starts 8-byte aligned. B blocks of one K chunk share a stride so a
REPEATed GEMM with post-increment walks them; C keeps one partial plane per K
chunk and the CPU reduces the planes (and applies per-chunk scales, which is
how group-wise INT4 weights stay exact on an INT8 array).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import isa


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
        return isa.align(self.m_i(mi) * self.tk)

    def a_offset(self, mi: int, kc: int) -> int:
        # Full tile rows precede row mi; within a row, chunks share a stride.
        return mi * self.kt * isa.align(self.tm * self.tk) + kc * self.a_tile_stride(mi)

    @property
    def a_bytes(self) -> int:
        last = self.mt - 1
        return self.a_offset(last, self.kt - 1) + isa.align(self.m_i(last) * self.k_c(self.kt - 1))

    # -- B ----------------------------------------------------------------
    def b_tile_stride(self, kc: int) -> int:
        return isa.align(self.k_c(kc) * self.tn)

    def b_offset(self, kc: int, ni: int) -> int:
        return kc * self.nt * isa.align(self.tk * self.tn) + ni * self.b_tile_stride(kc)

    @property
    def b_bytes(self) -> int:
        return self.b_offset(self.kt - 1, self.nt - 1) + self.b_tile_stride(self.kt - 1)

    # -- C ----------------------------------------------------------------
    def c_tile_stride(self, mi: int) -> int:
        return isa.align(4 * self.m_i(mi) * self.tn)

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


def plan_gemm(m: int, n: int, k: int, limits: isa.Limits = isa.Limits(), tk: int | None = None) -> GemmPlan:
    if min(m, n, k) < 1:
        raise isa.ISAError("GEMM dimensions must be positive")
    tk = limits.max_k if tk is None else tk
    if not 1 <= tk <= limits.max_k:
        raise isa.ISAError(f"K chunk {tk} exceeds MAX_K {limits.max_k}")
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


def encode(plan: GemmPlan, a_base: int, b_base: int, c_base: int) -> list[int]:
    """Instruction words (without END) computing every job of ``plan``."""
    words: list[int] = []
    for mi in range(plan.mt):
        for kc in range(plan.kt):
            full = plan.nt if plan.n_i(plan.nt - 1) == plan.tn else plan.nt - 1
            words.append(isa.addr_a(a_base + plan.a_offset(mi, kc)))
            words.append(isa.addr_b(b_base + plan.b_offset(kc, 0)))
            words.append(isa.addr_c(c_base + plan.c_offset(kc, mi, 0)))
            words.append(isa.incr(0, plan.b_tile_stride(kc), plan.c_tile_stride(mi)))
            if full:
                words.append(isa.shape(plan.m_i(mi), plan.tn, plan.k_c(kc)))
                if full > 1:
                    words.append(isa.repeat(full))
                words.append(isa.gemm(isa.GEMM_INC_B | isa.GEMM_INC_C))
            if full < plan.nt:
                words.append(isa.shape(plan.m_i(mi), plan.n_i(plan.nt - 1), plan.k_c(kc)))
                words.append(isa.gemm(0))
    return words
