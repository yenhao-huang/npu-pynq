"""Golden numeric model for the Hexagon-style NPU units (issue #136).

Every floating-point function works on raw IEEE 754 bit patterns held in
Python ints, so the model compares directly with RTL signals:

* binary16 add/sub/mul: round-to-nearest-even, subnormal inputs and outputs,
  overflow to +-Inf, every NaN result is the canonical quiet NaN ``0x7E00``.
* binary32 add (tensor FP16-mode accumulation only): same rules, canonical
  NaN ``0x7FC00000``.

numpy gives correctly rounded results here: an FP16 x FP16 product is exact
in float32 (22-bit significand, exponent >= -48) and is rounded once to
float16; an FP16 + FP16 sum is exact in float64 (at most 51 significant bits)
and is rounded once; float32 + float32 in numpy is a single IEEE operation.

Integer lane helpers follow the vector-unit contract: ADD/SUB saturate, MUL
wraps, SRA rounds half up and saturates.  The ``pe_mp_*`` helpers model the
multi-precision tensor PE (``npu_pe_mp``) accumulation exactly, including the
accumulation order used by the hardware.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

FP16_CANONICAL_NAN = 0x7E00
FP32_CANONICAL_NAN = 0x7FC00000
FP16_POS_INF = 0x7C00
FP32_POS_INF = 0x7F800000

PREC_INT8 = 0
PREC_INT16 = 1
PREC_FP16 = 2

ACC_BITS = 48
ACC_MASK = (1 << ACC_BITS) - 1


# --------------------------------------------------------------------------
# bit-pattern conversion helpers
# --------------------------------------------------------------------------
def fp16_from_bits(bits: int) -> np.float16:
    return np.array([bits & 0xFFFF], dtype=np.uint16).view(np.float16)[0]


def fp16_to_bits(value) -> int:
    return int(np.array([value], dtype=np.float16).view(np.uint16)[0])


def fp32_from_bits(bits: int) -> np.float32:
    return np.array([bits & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]


def fp32_to_bits(value) -> int:
    return int(np.array([value], dtype=np.float32).view(np.uint32)[0])


def fp16_is_nan(bits: int) -> bool:
    return (bits & 0x7C00) == 0x7C00 and (bits & 0x03FF) != 0


def fp32_is_nan(bits: int) -> bool:
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def _canon16(bits: int) -> int:
    return FP16_CANONICAL_NAN if fp16_is_nan(bits) else bits


def _canon32(bits: int) -> int:
    return FP32_CANONICAL_NAN if fp32_is_nan(bits) else bits


# --------------------------------------------------------------------------
# floating point operations on bit patterns
# --------------------------------------------------------------------------
def fp16_mul(a: int, b: int) -> int:
    """binary16 a * b, RNE, canonical NaN."""
    with np.errstate(all="ignore"):
        exact = np.float32(fp16_from_bits(a)) * np.float32(fp16_from_bits(b))
        return _canon16(fp16_to_bits(np.float16(exact)))


def fp16_add(a: int, b: int, sub: bool = False) -> int:
    """binary16 a + b (a - b when ``sub``), RNE, canonical NaN."""
    if sub:
        b ^= 0x8000
    with np.errstate(all="ignore"):
        exact = np.float64(fp16_from_bits(a)) + np.float64(fp16_from_bits(b))
        return _canon16(fp16_to_bits(np.float16(exact)))


def fp32_add(a: int, b: int, sub: bool = False) -> int:
    """binary32 a + b (a - b when ``sub``), RNE, canonical NaN."""
    if sub:
        b ^= 0x80000000
    with np.errstate(all="ignore"):
        result = fp32_from_bits(a) + fp32_from_bits(b)
        return _canon32(fp32_to_bits(result))


def fp16_mul_to_fp32(a: int, b: int) -> int:
    """Exact binary16 x binary16 product as binary32 bits (no rounding)."""
    with np.errstate(all="ignore"):
        exact = np.float32(fp16_from_bits(a)) * np.float32(fp16_from_bits(b))
        return _canon32(fp32_to_bits(exact))


def fp16_max(a: int, b: int) -> int:
    """NaN in -> 0x7E00; max(-0, +0) = +0."""
    if fp16_is_nan(a) or fp16_is_nan(b):
        return FP16_CANONICAL_NAN
    fa, fb = float(fp16_from_bits(a)), float(fp16_from_bits(b))
    if fa == fb:
        return a & b  # equal values: only +-0 differ, and +0 wins (sign 0)
    return a if fa > fb else b


def fp16_min(a: int, b: int) -> int:
    """NaN in -> 0x7E00; min(-0, +0) = -0."""
    if fp16_is_nan(a) or fp16_is_nan(b):
        return FP16_CANONICAL_NAN
    fa, fb = float(fp16_from_bits(a)), float(fp16_from_bits(b))
    if fa == fb:
        return a | b  # equal values: only +-0 differ, and -0 wins (sign 1)
    return a if fa < fb else b


# --------------------------------------------------------------------------
# integer lane helpers
# --------------------------------------------------------------------------
def int_sat(x: int, bits: int) -> int:
    """Saturate a Python int to the signed ``bits``-wide range."""
    low = -(1 << (bits - 1))
    high = (1 << (bits - 1)) - 1
    return max(low, min(high, x))


def int_wrap(x: int, bits: int) -> int:
    """Two's-complement wrap to ``bits`` (MUL keeps the low half)."""
    x &= (1 << bits) - 1
    return x - (1 << bits) if x >> (bits - 1) else x


def sra_round(x: int, s: int, bits: int) -> int:
    """Arithmetic shift right with round-half-up, then saturate."""
    if s > 0:
        x = (x + (1 << (s - 1))) >> s
    return int_sat(x, bits)


def to_signed(x: int, bits: int) -> int:
    return int_wrap(x, bits)


# --------------------------------------------------------------------------
# multi-precision tensor PE (npu_pe_mp) model
# --------------------------------------------------------------------------
def pe_mp_product_int(prec: int, a: int, b: int) -> int:
    """Signed integer product of two 16-bit operand lanes."""
    if prec == PREC_INT8:
        return to_signed(a & 0xFF, 8) * to_signed(b & 0xFF, 8)
    if prec == PREC_INT16:
        return to_signed(a & 0xFFFF, 16) * to_signed(b & 0xFFFF, 16)
    raise ValueError("integer precision expected")


def pe_mp_accumulate(prec: int, pairs: Iterable[Sequence[int]], acc: int = 0) -> int:
    """Accumulate (a, b) operand pairs in order; returns accumulator[47:0].

    INT8: 32-bit saturating accumulator (npu_pe semantics), sign-extended.
    INT16: 48-bit saturating accumulator.
    FP16: acc = fp32_add(acc, fp16_mul_to_fp32(a, b)) starting from +0,
    result in bits [31:0], upper bits zero.
    ``acc`` is the starting accumulator as raw 48-bit bits.
    """
    if prec == PREC_FP16:
        value = acc & 0xFFFFFFFF
        for a, b in pairs:
            value = fp32_add(value, fp16_mul_to_fp32(a, b))
        return value
    bits = 32 if prec == PREC_INT8 else 48
    value = to_signed(acc & ACC_MASK, ACC_BITS)
    for a, b in pairs:
        value = int_sat(value + pe_mp_product_int(prec, a, b), bits)
    return value & ACC_MASK


def gemm_mp(prec: int, matrix_a, matrix_b):
    """C[i][j] = PE accumulation of A[i][k] * B[k][j] for k = 0..K-1."""
    rows = len(matrix_a)
    depth = len(matrix_b)
    columns = len(matrix_b[0]) if depth else 0
    return [
        [
            pe_mp_accumulate(
                prec, ((matrix_a[i][k], matrix_b[k][j]) for k in range(depth))
            )
            for j in range(columns)
        ]
        for i in range(rows)
    ]


__all__ = [
    "FP16_CANONICAL_NAN",
    "FP32_CANONICAL_NAN",
    "PREC_INT8",
    "PREC_INT16",
    "PREC_FP16",
    "fp16_from_bits",
    "fp16_to_bits",
    "fp32_from_bits",
    "fp32_to_bits",
    "fp16_mul",
    "fp16_add",
    "fp32_add",
    "fp16_mul_to_fp32",
    "fp16_max",
    "fp16_min",
    "int_sat",
    "int_wrap",
    "sra_round",
    "pe_mp_accumulate",
    "gemm_mp",
]
