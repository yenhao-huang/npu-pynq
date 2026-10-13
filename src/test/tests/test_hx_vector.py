"""Checks for the vector-unit reference model (src/test/model/hx_vector.py).

Lane semantics are checked against hand-computed cases and against
independent implementations written here (exact Fraction rounding for SRA,
float comparisons with explicit signed-zero / NaN rules for FP16 MAX/MIN),
FP16 ADD/SUB/MUL lanes against :mod:`hx_numeric`, the architectural model
(VLD/VST/VSPLAT/VMOV, illegal commands) on small programs, and the committed
``vector_unit_programs.txt`` / ``vector_unit_programs_v128.txt`` are checked to
regenerate byte-identically.
"""

import math
import random
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from src.test.model import hx_numeric as hx
from src.test.model import hx_vector as hv
from src.test.model.hx_vector import VectorCommand as Cmd
from src.test.vectors import generate_vector_unit_vectors as gen
from src.test.vectors.generate_hx_numeric_vectors import FP16_CORNERS

VECTOR_DIR = Path("src/test/vectors")
NAN = 0x7E00
POS_ZERO, NEG_ZERO = 0x0000, 0x8000


def lanes(word, bits):
    return hv.split_lanes(word, bits, 64)


def word(values, bits):
    return hv.join_lanes(list(values), bits)


# --------------------------------------------------------------------------
# independent references
# --------------------------------------------------------------------------
def ref_sra(x, s, bits):
    """floor(x / 2^s + 1/2) (round half up), saturated; s = 0 is a copy."""
    if s == 0:
        r = x
    else:
        q = Fraction(x, 1 << s) + Fraction(1, 2)
        r = q.numerator // q.denominator
    return max(-(1 << (bits - 1)), min((1 << (bits - 1)) - 1, r))


def ref_fp16_minmax(a, b, want_max):
    if hx.fp16_is_nan(a) or hx.fp16_is_nan(b):
        return NAN
    fa, fb = float(hx.fp16_from_bits(a)), float(hx.fp16_from_bits(b))
    if fa == fb:
        # only +0 / -0 compare equal with different bits
        if fa == 0.0 and (a != b):
            return POS_ZERO if want_max else NEG_ZERO
        return a
    if want_max:
        return a if fa > fb else b
    return a if fa < fb else b


def signed(x, bits):
    return x - (1 << bits) if x >> (bits - 1) else x


# --------------------------------------------------------------------------
# integer lanes
# --------------------------------------------------------------------------
class IntegerLaneTests(unittest.TestCase):
    def test_add_sub_saturate_hand_cases(self):
        cases = [
            # (op, bits, a, b, expected)  raw lane bits
            (hv.OP_VADD, 8, 0x7F, 0x01, 0x7F),
            (hv.OP_VADD, 8, 0x80, 0xFF, 0x80),
            (hv.OP_VADD, 8, 0x7F, 0x80, 0xFF),
            (hv.OP_VSUB, 8, 0x80, 0x01, 0x80),
            (hv.OP_VSUB, 8, 0x7F, 0xFF, 0x7F),
            (hv.OP_VSUB, 8, 0x00, 0x80, 0x7F),
            (hv.OP_VADD, 16, 0x7FFF, 0x7FFF, 0x7FFF),
            (hv.OP_VSUB, 16, 0x8000, 0x7FFF, 0x8000),
            (hv.OP_VADD, 32, 0x7FFFFFFF, 0x00000001, 0x7FFFFFFF),
            (hv.OP_VSUB, 32, 0x80000000, 0x00000001, 0x80000000),
            (hv.OP_VADD, 32, 0xFFFFFFFF, 0x00000001, 0x00000000),
        ]
        typ = {8: hv.TYPE_INT8, 16: hv.TYPE_INT16, 32: hv.TYPE_INT32}
        for op, bits, a, b, expected in cases:
            with self.subTest(op=op, bits=bits, a=hex(a), b=hex(b)):
                self.assertEqual(hv.int_lane_op(op, a, b, bits, 0), expected)
                # the same lane in every position of a 64-bit word
                n = 64 // bits
                got = hv.vector_op(op, typ[bits], word([a] * n, bits), word([b] * n, bits), 0)
                self.assertEqual(got, word([expected] * n, bits))

    def test_mul_wraps_and_max_min_relu_are_signed(self):
        self.assertEqual(hv.int_lane_op(hv.OP_VMUL, 0x80, 0xFF, 8, 0), 0x80)   # -128 * -1
        self.assertEqual(hv.int_lane_op(hv.OP_VMUL, 0x10, 0x10, 8, 0), 0x00)   # 256 wraps
        self.assertEqual(hv.int_lane_op(hv.OP_VMUL, 0xFFFF, 0xFFFF, 16, 0), 0x0001)
        self.assertEqual(hv.int_lane_op(hv.OP_VMUL, 0x00010001, 0x00010001, 32, 0), 0x00020001)
        self.assertEqual(hv.int_lane_op(hv.OP_VMAX, 0x80, 0x7F, 8, 0), 0x7F)
        self.assertEqual(hv.int_lane_op(hv.OP_VMIN, 0x80, 0x7F, 8, 0), 0x80)
        self.assertEqual(hv.int_lane_op(hv.OP_VMAX, 0xFFFF, 0x0000, 16, 0), 0x0000)
        self.assertEqual(hv.int_lane_op(hv.OP_VRELU, 0x80000000, 0, 32, 0), 0)
        self.assertEqual(hv.int_lane_op(hv.OP_VRELU, 0x7FFFFFFF, 0x80000000, 32, 0), 0x7FFFFFFF)

    def test_sra_rounds_half_up_hand_cases(self):
        cases = [
            # (bits, x, s, expected signed result)
            (8, 5, 1, 3), (8, -5, 1, -2), (8, 6, 2, 2), (8, -6, 2, -1),
            (8, 3, 1, 2), (8, -3, 1, -1), (8, -1, 1, 0), (8, 1, 1, 1),
            (8, 127, 1, 64), (8, -128, 1, -64), (8, 127, 7, 1), (8, -128, 7, -1),
            (8, 127, 8, 0), (8, -128, 8, 0), (8, -128, 31, 0), (8, 64, 0, 64),
            (16, 0x7FFF, 15, 1), (16, -0x8000, 15, -1), (16, -0x8000, 16, 0),
            (32, 0x7FFFFFFF, 31, 1), (32, -0x80000000, 31, -1), (32, -0x80000000, 30, -2),
        ]
        for bits, x, s, expected in cases:
            with self.subTest(bits=bits, x=x, s=s):
                raw = x & ((1 << bits) - 1)
                got = hv.int_lane_op(hv.OP_VSRA, raw, 0, bits, s)
                self.assertEqual(signed(got, bits), expected)

    def test_sra_matches_exact_reference_and_ignores_upper_scalar_bits(self):
        rng = random.Random(0x5EA)
        for bits in (8, 16, 32):
            values = [0, 1, -1, 2, -2, 3, -3, (1 << (bits - 1)) - 1, -(1 << (bits - 1))]
            values += [rng.randrange(-(1 << (bits - 1)), 1 << (bits - 1)) for _ in range(200)]
            for x in values:
                for s in range(32):
                    scalar = (rng.getrandbits(27) << 5) | s
                    got = hv.int_lane_op(hv.OP_VSRA, x & ((1 << bits) - 1), 0, bits, scalar)
                    self.assertEqual(signed(got, bits), ref_sra(x, s, bits), (bits, x, s))
                    # also equals the golden numeric helper
                    self.assertEqual(signed(got, bits), hx.sra_round(x, s, bits))

    def test_random_lanes_match_python_integers(self):
        rng = random.Random(0x1A7E)
        for typ, bits in ((hv.TYPE_INT8, 8), (hv.TYPE_INT16, 16), (hv.TYPE_INT32, 32)):
            lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
            for _ in range(300):
                a, b = rng.getrandbits(64), rng.getrandbits(64)
                for op in (hv.OP_VADD, hv.OP_VSUB, hv.OP_VMUL, hv.OP_VMAX, hv.OP_VMIN,
                           hv.OP_VRELU):
                    got = lanes(hv.vector_op(op, typ, a, b, 0), bits)
                    for i, (x, y) in enumerate(zip(lanes(a, bits), lanes(b, bits))):
                        sx, sy = signed(x, bits), signed(y, bits)
                        exp = {
                            hv.OP_VADD: max(lo, min(hi, sx + sy)),
                            hv.OP_VSUB: max(lo, min(hi, sx - sy)),
                            hv.OP_VMUL: signed((sx * sy) & ((1 << bits) - 1), bits),
                            hv.OP_VMAX: max(sx, sy),
                            hv.OP_VMIN: min(sx, sy),
                            hv.OP_VRELU: max(sx, 0),
                        }[op]
                        self.assertEqual(signed(got[i], bits), exp, (op, typ, hex(a), hex(b)))


# --------------------------------------------------------------------------
# FP16 lanes
# --------------------------------------------------------------------------
class Fp16LaneTests(unittest.TestCase):
    def test_signed_zero_and_nan_rules(self):
        self.assertEqual(hv.fp16_lane_op(hv.OP_VMAX, NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(hv.fp16_lane_op(hv.OP_VMAX, POS_ZERO, NEG_ZERO), POS_ZERO)
        self.assertEqual(hv.fp16_lane_op(hv.OP_VMIN, NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(hv.fp16_lane_op(hv.OP_VMIN, POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(hv.fp16_lane_op(hv.OP_VRELU, NEG_ZERO, 0x1234), POS_ZERO)
        self.assertEqual(hv.fp16_lane_op(hv.OP_VRELU, 0xBC00, 0x1234), POS_ZERO)   # -1.0
        self.assertEqual(hv.fp16_lane_op(hv.OP_VRELU, 0x3C00, 0x1234), 0x3C00)
        self.assertEqual(hv.fp16_lane_op(hv.OP_VRELU, 0xFC00, 0x1234), POS_ZERO)   # -Inf
        for nan in (0x7C01, 0x7E00, 0x7FFF, 0xFC01, 0xFE00):
            self.assertEqual(hv.fp16_lane_op(hv.OP_VRELU, nan, 0), NAN)
            for other in (POS_ZERO, NEG_ZERO, 0x3C00, 0x7C00, 0xFC00, nan):
                for op in (hv.OP_VMAX, hv.OP_VMIN):
                    self.assertEqual(hv.fp16_lane_op(op, nan, other), NAN)
                    self.assertEqual(hv.fp16_lane_op(op, other, nan), NAN)
        self.assertEqual(hv.fp16_lane_op(hv.OP_VMAX, 0xFC00, 0x7C00), 0x7C00)
        self.assertEqual(hv.fp16_lane_op(hv.OP_VMIN, 0x0001, 0x8001), 0x8001)

    def test_max_min_match_independent_reference(self):
        rng = random.Random(0xF16)
        pairs = [(a, b) for a in FP16_CORNERS for b in FP16_CORNERS]
        pairs += [(rng.getrandbits(16), rng.getrandbits(16)) for _ in range(20000)]
        for a, b in pairs:
            self.assertEqual(hv.fp16_lane_op(hv.OP_VMAX, a, b), ref_fp16_minmax(a, b, True),
                             (hex(a), hex(b)))
            self.assertEqual(hv.fp16_lane_op(hv.OP_VMIN, a, b), ref_fp16_minmax(a, b, False),
                             (hex(a), hex(b)))

    def test_rtl_compare_identity(self):
        """npu_vector_alu orders FP16 by the INT16 signed compare of the raw
        bits XOR (sign_a & sign_b); check that this selects fp16_max/min."""
        rng = random.Random(0xC0)
        pairs = [(a, b) for a in FP16_CORNERS for b in FP16_CORNERS]
        pairs += [(rng.getrandbits(16), rng.getrandbits(16)) for _ in range(20000)]
        for a, b in pairs:
            if hx.fp16_is_nan(a) or hx.fp16_is_nan(b):
                continue
            lt = (signed(a, 16) < signed(b, 16)) ^ bool((a >> 15) & (b >> 15))
            self.assertEqual(b if lt else a, hx.fp16_max(a, b), (hex(a), hex(b)))
            self.assertEqual(a if lt else b, hx.fp16_min(a, b), (hex(a), hex(b)))

    def test_add_sub_mul_lanes_use_golden_numeric(self):
        rng = random.Random(0xADD)
        for _ in range(500):
            a = word([rng.choice(FP16_CORNERS) if rng.random() < 0.4 else rng.getrandbits(16)
                      for _ in range(4)], 16)
            b = word([rng.choice(FP16_CORNERS) if rng.random() < 0.4 else rng.getrandbits(16)
                      for _ in range(4)], 16)
            for op, fn in ((hv.OP_VADD, lambda x, y: hx.fp16_add(x, y)),
                           (hv.OP_VSUB, lambda x, y: hx.fp16_add(x, y, sub=True)),
                           (hv.OP_VMUL, hx.fp16_mul)):
                got = hv.vector_op(op, hv.TYPE_FP16, a, b, 0)
                exp = word([fn(x, y) for x, y in zip(lanes(a, 16), lanes(b, 16))], 16)
                self.assertEqual(got, exp)
        # canonical NaN and overflow
        self.assertEqual(hv.fp16_lane_op(hv.OP_VSUB, 0x7C00, 0x7C00), NAN)
        self.assertEqual(hv.fp16_lane_op(hv.OP_VMUL, 0x0000, 0x7C00), NAN)
        self.assertEqual(hv.fp16_lane_op(hv.OP_VADD, 0x7BFF, 0x7BFF), 0x7C00)
        self.assertTrue(math.isinf(float(hx.fp16_from_bits(0x7C00))))


# --------------------------------------------------------------------------
# architectural model
# --------------------------------------------------------------------------
class ModelTests(unittest.TestCase):
    def test_load_store_splat_mov(self):
        m = hv.VectorUnitModel()
        m.tcm[5] = 0x0123456789ABCDEF
        m.run([
            Cmd(hv.OP_VLD, 3, vd=1, addr=5),
            Cmd(hv.OP_VMOV, 2, vd=2, vs1=1),
            Cmd(hv.OP_VST, 0, vs1=2, addr=1023),
            Cmd(hv.OP_VSPLAT, hv.TYPE_INT8, vd=3, scalar=0x123456F0),
            Cmd(hv.OP_VSPLAT, hv.TYPE_INT16, vd=4, scalar=0x12348001),
            Cmd(hv.OP_VSPLAT, hv.TYPE_INT32, vd=5, scalar=0x89ABCDEF),
            Cmd(hv.OP_VSPLAT, hv.TYPE_FP16, vd=6, scalar=0xFFFF3C00),
        ])
        self.assertEqual(m.rf[1], 0x0123456789ABCDEF)
        self.assertEqual(m.rf[2], 0x0123456789ABCDEF)
        self.assertEqual(m.tcm[1023], 0x0123456789ABCDEF)
        self.assertEqual(m.rf[3], 0xF0F0F0F0F0F0F0F0)
        self.assertEqual(m.rf[4], 0x8001800180018001)
        self.assertEqual(m.rf[5], 0x89ABCDEF89ABCDEF)
        self.assertEqual(m.rf[6], 0x3C003C003C003C00)
        self.assertFalse(m.illegal)

    def test_illegal_commands_change_nothing_but_the_flag(self):
        m = hv.VectorUnitModel()
        m.rf = [(0x1111111111111111 * (r + 1)) & m.mask for r in range(8)]
        m.tcm[7] = 0xDEAD
        rf0, tcm0 = list(m.rf), list(m.tcm)
        for op in range(11, 32):
            m.execute(Cmd(op, 0, vd=1, vs1=2, vs2=3, addr=7))
        m.execute(Cmd(hv.OP_VSRA, hv.TYPE_FP16, vd=1, vs1=2, scalar=1))
        self.assertTrue(m.illegal)
        self.assertEqual(m.rf, rf0)
        self.assertEqual(m.tcm, tcm0)
        self.assertTrue(hv.is_legal(hv.OP_VSRA, hv.TYPE_INT32))
        self.assertFalse(hv.is_legal(hv.OP_VSRA, hv.TYPE_FP16))

    def test_relu_ignores_vs2(self):
        m = hv.VectorUnitModel()
        m.rf[1] = word([0x80, 0x7F, 0x00, 0xFF, 0x01, 0x81, 0x40, 0xC0], 8)
        m.rf[2] = 0xFFFFFFFFFFFFFFFF
        m.execute(Cmd(hv.OP_VRELU, hv.TYPE_INT8, vd=3, vs1=1, vs2=2))
        self.assertEqual(m.rf[3], word([0x00, 0x7F, 0x00, 0x00, 0x01, 0x00, 0x40, 0x00], 8))


    def test_vlen128_is_two_independent_vlen64_halves(self):
        # the VLEN = 128 testbench instance relies on lanes never crossing a
        # 64-bit boundary: the model at 128 bits equals two 64-bit halves
        rng = random.Random(128)
        m64 = (1 << 64) - 1
        ops = (hv.OP_VADD, hv.OP_VSUB, hv.OP_VMUL, hv.OP_VMAX, hv.OP_VMIN, hv.OP_VRELU,
               hv.OP_VSRA)
        for op in ops:
            for typ in range(4):
                if not hv.is_legal(op, typ):
                    continue
                for _ in range(50):
                    a, b, sc = rng.getrandbits(128), rng.getrandbits(128), rng.getrandbits(32)
                    full = hv.vector_op(op, typ, a, b, sc, 128)
                    lo = hv.vector_op(op, typ, a & m64, b & m64, sc, 64)
                    hi = hv.vector_op(op, typ, a >> 64, b >> 64, sc, 64)
                    self.assertEqual(full, lo | (hi << 64), (op, typ, hex(a), hex(b)))


class VectorFileTests(unittest.TestCase):
    def test_pre_field_only_in_paced_kernels(self):
        programs = gen.build()
        kernels = [p for p in programs if p.mode & gen.MODE_PACED]
        self.assertEqual(len(kernels), 6)
        for prog in programs:
            self.assertEqual(len(prog.pre), len(prog.commands), prog.name)
            if not prog.mode & gen.MODE_PACED:
                self.assertFalse(any(prog.pre), prog.name)
        for prog in kernels:
            self.assertEqual(prog.perf_start, 8, prog.name)   # after the register init
            expect = {"kern_res8_loop": 5, "kern_axpy16_loop": 4}.get(prog.name, 0)
            self.assertEqual(max(prog.pre), expect, prog.name)
        for prog in gen.build_v128():
            self.assertEqual(prog.vlen, 128)
            self.assertFalse(prog.mode & (gen.MODE_PACED | gen.MODE_PERF), prog.name)

    def test_committed_programs_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            gen.write_all(Path(directory))
            for name in (gen.OUTPUT_FILE, gen.OUTPUT_FILE_V128):
                self.assertEqual(
                    (Path(directory) / name).read_bytes(),
                    (VECTOR_DIR / name).read_bytes(),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
