"""Generate reproducible vectors for the Hexagon numeric and tensor_mp units.

Outputs (all in this directory, committed):

* ``hx_fp16_mul.txt``  ``a b y``            (hex)
* ``hx_fp16_add.txt``  ``a b sub y``        (hex, sub is 0/1)
* ``hx_fp32_add.txt``  ``a b sub y``        (hex)
* ``hx_pe_mp.txt``     per case ``prec k`` then k lines ``a b`` then the
                       expected 48-bit accumulator (hex)
* ``hx_systolic_mp_4x4.txt``  per case ``prec k``, A (4 x k, row-major),
                       B (k x 4, row-major), expected C (4 x 4, 48-bit hex)

Every file starts with one ``#`` comment line and a decimal case count.
Run from the repository root:
``python3 -m src.test.vectors.generate_hx_numeric_vectors``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.test.model import hx_numeric as hx

DEFAULT_SEED = 0x136F
RANDOM_FP16 = 20000
RANDOM_FP32 = 20000
PE_CASES = 120
ARRAY_CASES = 60
ARRAY_ROWS = 4
ARRAY_COLUMNS = 4
ARRAY_MAX_K = 12
PE_MAX_K = 16

OUTPUT_FILES = (
    "hx_fp16_mul.txt",
    "hx_fp16_add.txt",
    "hx_fp32_add.txt",
    "hx_pe_mp.txt",
    "hx_systolic_mp_4x4.txt",
)

FP16_CORNERS = (
    0x0000, 0x8000,           # signed zeros
    0x0001, 0x8001, 0x0002, 0x0003, 0x0200,  # subnormals
    0x03FF, 0x83FF,           # max subnormal
    0x0400, 0x8400, 0x0401, 0x07FF,  # min normal and neighbours
    0x0C00, 0x1000, 0x1400,   # 2^-12, 2^-11, 2^-10 (halfway partners for 1.0)
    0x3555, 0x3800, 0x3BFF, 0x3C00, 0xBC00, 0x3C01, 0x3E00, 0x4000,
    0x5640, 0x6800,
    0x7BFE, 0x7BFF, 0xFBFF,   # max normal
    0x7C00, 0xFC00,           # infinities
    0x7E00, 0x7C01, 0xFE00, 0x7FFF,  # NaNs (quiet, signalling, negative)
)

FP32_CORNERS = (
    0x00000000, 0x80000000,
    0x00000001, 0x80000001, 0x00000002, 0x00000003, 0x00400000,
    0x007FFFFF, 0x807FFFFF,
    0x00800000, 0x80800000, 0x00800001, 0x00FFFFFF,
    0x33000000, 0x33800000, 0x34000000,  # 2^-25, 2^-24, 2^-23
    0x3EAAAAAB, 0x3F000000, 0x3F7FFFFF, 0x3F800000, 0xBF800000, 0x3F800001,
    0x40490FDB, 0x4B000000, 0x4B800000,
    0x7F7FFFFE, 0x7F7FFFFF, 0xFF7FFFFF,
    0x7F800000, 0xFF800000,
    0x7FC00000, 0x7F800001, 0xFFFFFFFF,
)


def corner_pairs(corners):
    return [(a, b) for a in corners for b in corners]


# --------------------------------------------------------------------------
# random operand generators
# --------------------------------------------------------------------------
def _fp_fields(ebits, mbits):
    return (1 << ebits) - 1, (1 << mbits) - 1, ebits + mbits


def _random_value(rng, ebits, mbits, exp_low, exp_high):
    emax, mmask, sign_bit = _fp_fields(ebits, mbits)
    exponent = rng.randint(max(0, exp_low), min(emax - 1, exp_high))
    return (rng.getrandbits(1) << sign_bit) | (exponent << mbits) | (
        rng.getrandbits(mbits)
    )


def _nearby_pair(rng, ebits, mbits):
    """Two finite values whose exponents differ by at most a few steps."""
    emax, _, _ = _fp_fields(ebits, mbits)
    base = rng.randint(0, emax - 1)
    span = rng.choice((0, 0, 1, 2, 3, mbits + 1, mbits + 3))
    a = _random_value(rng, ebits, mbits, base, base)
    b = _random_value(rng, ebits, mbits, base - span, base + span)
    if rng.random() < 0.25:
        # near-total cancellation: same magnitude bits apart from a few LSBs
        b = (a ^ rng.getrandbits(3)) ^ (rng.getrandbits(1) << (ebits + mbits))
    return a, b


def _fp16_mul_operands(rng):
    """Values whose exponent sum lands anywhere incl. under/overflow."""
    a = _random_value(rng, 5, 10, 0, 30)
    b = _random_value(rng, 5, 10, 0, 30)
    if rng.random() < 0.3:
        # aim the product exponent at the subnormal/overflow boundaries
        target = rng.choice((-10, -2, 0, 1, 2, 29, 30, 31, 32))
        ea = (a >> 10) & 0x1F
        eb = target + 15 - ea
        if 0 <= eb <= 30:
            b = (b & 0x83FF) | (eb << 10)
    return a, b


def mul_sticky_cases():
    """Subnormal-result products that are exactly halfway above the kept
    bits except for bits shifted out by the denormalising right shift.

    Only the shifted-out sticky bits decide the rounding here (guard = 1,
    fraction LSB = 0, the ten bits below guard all zero), which random
    operands almost never hit. For each right shift r = 1..11 (none exist for
    r = 10) the first few normal significand pairs are taken and given
    exponents with ea + eb = 16 - r - (product >= 2), in both signs.
    """
    cases = []
    for shift in range(1, 12):
        found = 0
        for sig_a in (1025, 1057, 1279, 2047):
            for sig_b in range(1024, 2048):
                product = sig_a * sig_b
                high = product >> 21
                mant = product if high else product << 1   # leading one at bit 21
                kept = mant >> shift
                lost = mant & ((1 << shift) - 1)
                if not ((kept >> 10) & 1) or ((kept >> 11) & 1) or (kept & 0x3FF) or not lost:
                    continue
                exp_sum = 16 - shift - high
                exp_a = exp_sum // 2
                a = (exp_a << 10) | (sig_a - 1024)
                b = ((exp_sum - exp_a) << 10) | (sig_b - 1024)
                cases.extend([(a, b), (b | 0x8000, a)])
                found += 1
                if found == 3:
                    break
            if found == 3:
                break
    return cases


# --------------------------------------------------------------------------
# element-wise unit vectors
# --------------------------------------------------------------------------
def build_fp16_mul(rng):
    vectors = [(a, b) for a, b in corner_pairs(FP16_CORNERS)]
    vectors.extend(mul_sticky_cases())
    for index in range(RANDOM_FP16):
        if index % 2 == 0:
            vectors.append((rng.getrandbits(16), rng.getrandbits(16)))
        else:
            vectors.append(_fp16_mul_operands(rng))
    return [(a, b, hx.fp16_mul(a, b)) for a, b in vectors]


def build_add(rng, width, corners, random_count, model):
    ebits, mbits = (5, 10) if width == 16 else (8, 23)
    one = 0x3C00 if width == 16 else 0x3F800000
    vectors = []
    for a, b in corner_pairs(corners):
        vectors.append((a, b, 0))
        vectors.append((a, b, 1))
    # directed halfway cases around 1.0: 1 + ulp/2 (tie to even), 1+ulp + ulp/2
    half_ulp = (one - ((mbits + 1) << mbits))
    for a in (one, one + 1, one + 2, one + 3, one | (1 << (width - 1))):
        for b in (half_ulp, half_ulp + 1, half_ulp - 1):
            vectors.append((a, b, 0))
            vectors.append((a, b, 1))
    for index in range(random_count):
        if index % 2 == 0:
            a, b = rng.getrandbits(width), rng.getrandbits(width)
        else:
            a, b = _nearby_pair(rng, ebits, mbits)
        vectors.append((a, b, rng.getrandbits(1)))
    return [(a, b, s, model(a, b, bool(s))) for a, b, s in vectors]


# --------------------------------------------------------------------------
# tensor PE / array vectors
# --------------------------------------------------------------------------
INT8_ENDPOINTS = (-128, -127, -1, 0, 1, 127)
INT16_ENDPOINTS = (-32768, -32767, -1, 0, 1, 32767)
FP16_SPECIALS = (0x0000, 0x8000, 0x0001, 0x83FF, 0x7BFF, 0xFBFF, 0x7C00,
                 0xFC00, 0x7E00)


def _operand(rng, prec, special_rate=0.0):
    if prec == hx.PREC_INT8:
        if rng.random() < 0.2:
            low = rng.choice(INT8_ENDPOINTS) & 0xFF
        else:
            low = rng.getrandbits(8)
        return (rng.getrandbits(8) << 8) | low  # upper byte must be ignored
    if prec == hx.PREC_INT16:
        if rng.random() < 0.2:
            return rng.choice(INT16_ENDPOINTS) & 0xFFFF
        return rng.getrandbits(16)
    if rng.random() < special_rate:
        return rng.choice(FP16_SPECIALS)
    # moderate magnitudes so that partial sums interact (exp 2^-6 .. 2^5)
    return _random_value(rng, 5, 10, 9, 20)


def build_pe_cases(rng):
    cases = []
    for index in range(PE_CASES):
        prec = index % 3
        k = rng.randint(1, PE_MAX_K)
        special = 0.15 if (prec == hx.PREC_FP16 and index % 4 == 0) else 0.0
        pairs = [(_operand(rng, prec, special), _operand(rng, prec, special))
                 for _ in range(k)]
        cases.append((prec, pairs, hx.pe_mp_accumulate(prec, pairs)))
    return cases


def build_array_cases(rng):
    cases = []
    for index in range(ARRAY_CASES):
        prec = index % 3
        k = ARRAY_MAX_K if index < 3 else rng.randint(1, ARRAY_MAX_K)
        special = 0.05 if (prec == hx.PREC_FP16 and index % 2 == 0) else 0.0
        matrix_a = [[_operand(rng, prec, special) for _ in range(k)]
                    for _ in range(ARRAY_ROWS)]
        matrix_b = [[_operand(rng, prec, special) for _ in range(ARRAY_COLUMNS)]
                    for _ in range(k)]
        cases.append((prec, k, matrix_a, matrix_b,
                      hx.gemm_mp(prec, matrix_a, matrix_b)))
    return cases


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
def _write(path: Path, comment: str, count: int, lines):
    text = [f"# {comment}", str(count)] + list(lines)
    path.write_text("\n".join(text) + "\n", encoding="ascii", newline="\n")


def write_all(directory: Path, seed: int = DEFAULT_SEED) -> None:
    rng = random.Random(seed)

    mul = build_fp16_mul(rng)
    _write(directory / "hx_fp16_mul.txt",
           f"npu_fp16_mul vectors seed={seed}: a b y", len(mul),
           (f"{a:04x} {b:04x} {y:04x}" for a, b, y in mul))

    add16 = build_add(rng, 16, FP16_CORNERS, RANDOM_FP16, hx.fp16_add)
    _write(directory / "hx_fp16_add.txt",
           f"npu_fp16_add vectors seed={seed}: a b sub y", len(add16),
           (f"{a:04x} {b:04x} {s} {y:04x}" for a, b, s, y in add16))

    add32 = build_add(rng, 32, FP32_CORNERS, RANDOM_FP32, hx.fp32_add)
    _write(directory / "hx_fp32_add.txt",
           f"npu_fp32_add vectors seed={seed}: a b sub y", len(add32),
           (f"{a:08x} {b:08x} {s} {y:08x}" for a, b, s, y in add32))

    pe_lines = []
    pe_cases = build_pe_cases(rng)
    for prec, pairs, expected in pe_cases:
        pe_lines.append(f"{prec} {len(pairs)}")
        pe_lines.extend(f"{a:04x} {b:04x}" for a, b in pairs)
        pe_lines.append(f"{expected:012x}")
    _write(directory / "hx_pe_mp.txt",
           f"npu_pe_mp vectors seed={seed}: prec k, k x (a b), acc48",
           len(pe_cases), pe_lines)

    array_lines = []
    array_cases = build_array_cases(rng)
    for prec, k, matrix_a, matrix_b, matrix_c in array_cases:
        array_lines.append(f"{prec} {k}")
        array_lines.extend(" ".join(f"{v:04x}" for v in row) for row in matrix_a)
        array_lines.extend(" ".join(f"{v:04x}" for v in row) for row in matrix_b)
        array_lines.extend(" ".join(f"{v:012x}" for v in row) for row in matrix_c)
    _write(directory / "hx_systolic_mp_4x4.txt",
           f"npu_systolic_array_mp 4x4 vectors seed={seed}: prec k, A, B, C",
           len(array_cases), array_lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    write_all(arguments.output_dir, arguments.seed)


if __name__ == "__main__":
    main()
