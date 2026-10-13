"""Checks for the Hexagon numeric golden model (src/test/model/hx_numeric.py).

The numpy-based model is cross-checked against an independent exact
implementation (``fractions.Fraction`` plus an explicit round-to-nearest-even
encoder), and the committed vector files are checked to be byte-reproducible.
"""

import random
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from src.test.model import hx_numeric as hx
from src.test.vectors import generate_hx_numeric_vectors as gen

VECTOR_DIR = Path("src/test/vectors")


# --------------------------------------------------------------------------
# independent exact reference
# --------------------------------------------------------------------------
def _decode(bits, ebits, mbits):
    """Return ('nan'|'inf'|'num', sign, Fraction magnitude)."""
    sign = (bits >> (ebits + mbits)) & 1
    exp = (bits >> mbits) & ((1 << ebits) - 1)
    frac = bits & ((1 << mbits) - 1)
    bias = (1 << (ebits - 1)) - 1
    if exp == (1 << ebits) - 1:
        return ("nan" if frac else "inf"), sign, Fraction(0)
    if exp == 0:
        mag = Fraction(frac, 1 << mbits) * Fraction(2) ** (1 - bias)
    else:
        mag = Fraction((1 << mbits) + frac, 1 << mbits) * Fraction(2) ** (exp - bias)
    return "num", sign, mag


def _encode(value, zero_sign, ebits, mbits):
    """Round an exact Fraction to the format with RNE."""
    if value == 0:
        return zero_sign << (ebits + mbits)
    sign = 1 if value < 0 else 0
    mag = abs(value)
    bias = (1 << (ebits - 1)) - 1
    emin, emax = 1 - bias, bias
    exponent = mag.numerator.bit_length() - mag.denominator.bit_length()
    while Fraction(2) ** exponent > mag:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= mag:
        exponent += 1
    exponent = max(exponent, emin)
    scaled = mag * Fraction(2) ** (mbits - exponent)
    integer = scaled.numerator // scaled.denominator
    rest = scaled - integer
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and integer & 1):
        integer += 1
    if integer == 1 << (mbits + 1):
        integer >>= 1
        exponent += 1
    if exponent > emax:
        return (sign << (ebits + mbits)) | (((1 << ebits) - 1) << mbits)
    if integer < (1 << mbits):
        field = 0
    else:
        field = exponent + bias
        integer -= 1 << mbits
    return (sign << (ebits + mbits)) | (field << mbits) | integer


def _nan(ebits, mbits):
    return (((1 << ebits) - 1) << mbits) | (1 << (mbits - 1))


def ref_add(a, b, ebits, mbits, sub=False):
    if sub:
        b ^= 1 << (ebits + mbits)
    ka, sa, ma = _decode(a, ebits, mbits)
    kb, sb, mb = _decode(b, ebits, mbits)
    inf = ((1 << ebits) - 1) << mbits
    if ka == "nan" or kb == "nan" or (ka == kb == "inf" and sa != sb):
        return _nan(ebits, mbits)
    if ka == "inf":
        return (sa << (ebits + mbits)) | inf
    if kb == "inf":
        return (sb << (ebits + mbits)) | inf
    value = (-ma if sa else ma) + (-mb if sb else mb)
    zero_sign = 1 if (sa and sb) else 0
    return _encode(value, zero_sign, ebits, mbits)


def ref_mul(a, b, ebits, mbits, out_ebits=None, out_mbits=None):
    out_ebits = out_ebits or ebits
    out_mbits = out_mbits or mbits
    ka, sa, ma = _decode(a, ebits, mbits)
    kb, sb, mb = _decode(b, ebits, mbits)
    sign = sa ^ sb
    if ka == "nan" or kb == "nan":
        return _nan(out_ebits, out_mbits)
    if ka == "inf" or kb == "inf":
        if (ka == "num" and ma == 0) or (kb == "num" and mb == 0):
            return _nan(out_ebits, out_mbits)
        return (sign << (out_ebits + out_mbits)) | (((1 << out_ebits) - 1) << out_mbits)
    value = ma * mb
    return _encode(-value if sign else value, sign, out_ebits, out_mbits)


def _random_pairs(rng, width, count):
    pairs = []
    for _ in range(count):
        a = rng.getrandbits(width)
        if rng.random() < 0.5:
            b = rng.getrandbits(width)
        else:
            # nearby exponent: perturb a's exponent field slightly
            shift = 10 if width == 16 else 23
            b = a ^ (rng.getrandbits(shift + 3) | (rng.getrandbits(1) << (width - 1)))
        pairs.append((a, b))
    return pairs


class FloatModelTests(unittest.TestCase):
    def test_fp16_add_matches_exact_reference(self):
        rng = random.Random(11)
        for a, b in _random_pairs(rng, 16, 4000) + gen.corner_pairs(gen.FP16_CORNERS):
            for sub in (False, True):
                self.assertEqual(
                    hx.fp16_add(a, b, sub), ref_add(a, b, 5, 10, sub),
                    f"fp16_add({a:04x}, {b:04x}, sub={sub})",
                )

    def test_fp16_mul_matches_exact_reference(self):
        rng = random.Random(12)
        for a, b in _random_pairs(rng, 16, 4000) + gen.corner_pairs(gen.FP16_CORNERS):
            self.assertEqual(hx.fp16_mul(a, b), ref_mul(a, b, 5, 10),
                             f"fp16_mul({a:04x}, {b:04x})")
            self.assertEqual(hx.fp16_mul_to_fp32(a, b), ref_mul(a, b, 5, 10, 8, 23),
                             f"fp16_mul_to_fp32({a:04x}, {b:04x})")

    def test_fp32_add_matches_exact_reference(self):
        rng = random.Random(13)
        for a, b in _random_pairs(rng, 32, 4000) + gen.corner_pairs(gen.FP32_CORNERS):
            self.assertEqual(hx.fp32_add(a, b), ref_add(a, b, 8, 23),
                             f"fp32_add({a:08x}, {b:08x})")

    def test_mul_sticky_cases_depend_on_shifted_out_bits(self):
        # Each directed case is a subnormal result just above a tie with an
        # even LSB: RNE rounds up only because of the shifted-out sticky bits.
        cases = gen.mul_sticky_cases()
        self.assertGreaterEqual(len(cases), 20)
        for a, b in cases:
            _, _, mag_a = _decode(a, 5, 10)
            _, _, mag_b = _decode(b, 5, 10)
            units = mag_a * mag_b * 2 ** 24        # in min-subnormal units
            whole = units.numerator // units.denominator
            rest = units - whole
            self.assertEqual(whole % 2, 0, f"{a:04x} * {b:04x}")
            self.assertTrue(Fraction(1, 2) < rest < Fraction(1, 2) + Fraction(1, 2 ** 11),
                            f"{a:04x} * {b:04x}")
            self.assertEqual(hx.fp16_mul(a, b) & 0x7FFF, whole + 1, f"{a:04x} * {b:04x}")

    def test_ieee_corner_values(self):
        self.assertEqual(hx.fp16_mul(0x7C00, 0x0000), hx.FP16_CANONICAL_NAN)
        self.assertEqual(hx.fp16_add(0x7C00, 0x7C00, sub=True), hx.FP16_CANONICAL_NAN)
        self.assertEqual(hx.fp16_add(0x8000, 0x8000), 0x8000)
        self.assertEqual(hx.fp16_add(0x3C00, 0x3C00, sub=True), 0x0000)
        self.assertEqual(hx.fp16_add(0x7BFF, 0x7BFF), 0x7C00)
        self.assertEqual(hx.fp16_mul(0x0001, 0x3800), 0x0000)  # min/2 ties to even 0
        self.assertEqual(hx.fp16_mul(0x0003, 0x3800), 0x0002)  # 1.5 ulp ties to 2
        self.assertEqual(hx.fp32_add(0x7FC00001, 0x3F800000), hx.FP32_CANONICAL_NAN)
        self.assertEqual(hx.fp32_add(0x80000000, 0x80000000), 0x80000000)
        self.assertEqual(hx.fp16_max(0x8000, 0x0000), 0x0000)
        self.assertEqual(hx.fp16_min(0x8000, 0x0000), 0x8000)
        self.assertEqual(hx.fp16_max(0x7E01, 0x3C00), hx.FP16_CANONICAL_NAN)
        self.assertEqual(hx.fp16_min(0xBC00, 0x3C00), 0xBC00)


class IntegerModelTests(unittest.TestCase):
    def test_lane_helpers(self):
        self.assertEqual(hx.int_sat(200, 8), 127)
        self.assertEqual(hx.int_sat(-200, 8), -128)
        self.assertEqual(hx.int_wrap(0x1FF, 8), -1)
        self.assertEqual(hx.sra_round(5, 1, 8), 3)
        self.assertEqual(hx.sra_round(-5, 1, 8), -2)
        self.assertEqual(hx.sra_round(-7, 0, 8), -7)
        self.assertEqual(hx.sra_round(1 << 20, 2, 16), 32767)

    def test_pe_accumulation_saturates(self):
        pairs = [(0x7F, 0x7F)] * 3
        self.assertEqual(hx.pe_mp_accumulate(hx.PREC_INT8, pairs, acc=0x7FFFFFF0),
                         0x7FFFFFFF)
        self.assertEqual(hx.pe_mp_accumulate(hx.PREC_INT8, [(0xFF80, 0x017F)]),
                         (-128 * 127) & hx.ACC_MASK)
        self.assertEqual(hx.pe_mp_accumulate(hx.PREC_INT16, [(0x8000, 0x8000)],
                                             acc=0x7FFFFFFFFFFF), 0x7FFFFFFFFFFF)
        self.assertEqual(hx.pe_mp_accumulate(hx.PREC_FP16, [(0x3C00, 0x4000)] * 2),
                         0x40800000)


class VectorFileTests(unittest.TestCase):
    def test_committed_vectors_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            gen.write_all(Path(directory))
            for name in gen.OUTPUT_FILES:
                self.assertEqual(
                    (Path(directory) / name).read_bytes(),
                    (VECTOR_DIR / name).read_bytes(),
                    f"{name} is stale: run python3 -m "
                    "src.test.vectors.generate_hx_numeric_vectors",
                )


if __name__ == "__main__":
    unittest.main()
