"""Encodings of src/test/vectors/risc_v_npu/rv32asm.py against values from the
RISC-V unprivileged specification (and the encodings GNU as produces)."""

import unittest

from src.test.vectors.risc_v_npu.rv32asm import AsmError, Assembler, split_hi_lo


def one(build):
    asm = Assembler()
    build(asm)
    return asm.words()


class EncodingTest(unittest.TestCase):
    def test_single_words(self):
        cases = [
            (lambda a: a.nop(), 0x00000013),
            (lambda a: a.addi("x1", "x0", 1), 0x00100093),
            (lambda a: a.add("x3", "x1", "x2"), 0x002081B3),
            (lambda a: a.sub("x3", "x1", "x2"), 0x402081B3),
            (lambda a: a.and_("a0", "a1", "a2"), 0x00C5F533),
            (lambda a: a.or_("a0", "a1", "a2"), 0x00C5E533),
            (lambda a: a.lui("t0", 0x40000), 0x400002B7),
            (lambda a: a.auipc("ra", 0), 0x00000097),
            (lambda a: a.lw("t0", 0x1C, "s1"), 0x01C4A283),
            (lambda a: a.sw("zero", 0x20, "s1"), 0x0204A023),
            (lambda a: a.sw("a5", -4, "sp"), 0xFEF12E23),
            (lambda a: a.lw("a0", -4, "sp"), 0xFFC12503),
            (lambda a: a.jal("ra", 8), 0x008000EF),
            (lambda a: a.j(0), 0x0000006F),
            (lambda a: a.ret(), 0x00008067),
            (lambda a: a.jalr("ra", "a0", 0), 0x000500E7),
            (lambda a: a.beq("x0", "x0", 0), 0x00000063),
            (lambda a: a.bnez("t1", -8), 0xFE031CE3),
            (lambda a: a.blt("a0", "a1", 16), 0x00B54863),
            (lambda a: a.bge("a0", "a1", -16), 0xFEB558E3),
            (lambda a: a.mul("a0", "a0", "a1"), 0x02B50533),
            (lambda a: a.divu("a0", "a0", "a1"), 0x02B55533),
            (lambda a: a.slli("x1", "x1", 3), 0x00309093),
            (lambda a: a.srli("x1", "x1", 3), 0x0030D093),
            (lambda a: a.srai("x1", "x1", 3), 0x4030D093),
            (lambda a: a.ecall(), 0x00000073),
            (lambda a: a.ebreak(), 0x00100073),
            (lambda a: a.fence(), 0x0FF0000F),
            (lambda a: a.mv("a0", "a1"), 0x00058513),
        ]
        for build, want in cases:
            with self.subTest(want=hex(want)):
                self.assertEqual(one(build), [want])

    def test_li(self):
        self.assertEqual(one(lambda a: a.li("t0", 5)), [0x00500293])
        self.assertEqual(one(lambda a: a.li("t0", -1)), [0xFFF00293])
        self.assertEqual(one(lambda a: a.li("t0", 0x12345678)), [0x123452B7, 0x67828293])
        self.assertEqual(one(lambda a: a.li("t0", 0x800)), [0x000012B7, 0x80028293])
        self.assertEqual(one(lambda a: a.li("s1", 0x40000000)), [0x400004B7])
        self.assertEqual(one(lambda a: a.li("t0", 0xFFFFF800)), [0x80000293])
        self.assertEqual(one(lambda a: a.li("t0", 0x80000000)), [0x800002B7])

    def test_split_hi_lo(self):
        for value in (0, 1, 0x7FF, 0x800, 0xFFF, 0x1000, 0x12345678, 0x7FFFF800, 0xFFFFFFFF):
            hi, lo = split_hi_lo(value)
            self.assertEqual(((hi << 12) + lo) & 0xFFFFFFFF, value)
            self.assertTrue(-2048 <= lo < 2048)

    def test_labels_are_pc_relative(self):
        asm = Assembler()
        asm.li("t1", 3)
        asm.label("loop")
        asm.addi("t1", "t1", -1)
        asm.bnez("t1", "loop")
        asm.j("end")
        asm.nop()
        asm.label("end")
        asm.la("a0", "loop")
        words = asm.words()
        self.assertEqual(words[2], 0xFE031EE3)      # bnez t1, -4
        self.assertEqual(words[3], 0x0080006F)      # j +8
        self.assertEqual(words[5], 0x00000517)      # auipc a0, 0
        self.assertEqual(words[6], 0xFF050513)      # addi a0, a0, -16 (loop at 4, auipc at 20)

    def test_errors(self):
        with self.assertRaises(AsmError):
            one(lambda a: a.addi("t0", "t0", 4096))
        with self.assertRaises(AsmError):
            one(lambda a: a.j("nowhere"))
        with self.assertRaises(AsmError):
            one(lambda a: a.add("t0", "q7", "t1"))
        with self.assertRaises(AsmError):
            one(lambda a: a.beq("x0", "x0", 3))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# isa_to_rv: run each translated program on a small RV32I interpreter with a
# model of the MMIO job registers, and compare the issued jobs with isa_sim.

import numpy as np  # noqa: E402

from src.test.model import isa_sim  # noqa: E402
from src.test.vectors.risc_v_npu import isa_to_rv  # noqa: E402
from src.vendors.npu_pynq.spec import isa  # noqa: E402


def _sx(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def run_rv(code: bytes, data_base: int, limits=isa.Limits(16, 16, 256), budget=200000):
    """Returns (jobs as (m, n, k, a, b, c) absolute, end code)."""
    words = [int.from_bytes(code[i:i + 4], "little") for i in range(0, len(code), 4)]
    x = [0] * 32
    regs = {0x00: 0, 0x04: 0, 0x08: 0, 0x0C: 0, 0x10: 0}
    jobs = []
    pc = 0
    for _ in range(budget):
        w = words[pc // 4]
        op, rd, f3 = w & 0x7F, (w >> 7) & 31, (w >> 12) & 7
        rs1, rs2 = x[(w >> 15) & 31], x[(w >> 20) & 31]
        imm_i = _sx(w >> 20, 12)
        nxt = pc + 4
        val = None
        if op == 0x37:
            val = w & 0xFFFFF000
        elif op == 0x13 and f3 == 0:
            val = rs1 + imm_i
        elif op == 0x33 and w >> 25 == 0 and f3 == 0:
            val = rs1 + rs2
        elif op == 0x03 and f3 == 2:
            addr = (rs1 + imm_i) & 0xFFFFFFFF
            off = addr - isa_to_rv.MMIO_BASE
            val = {isa_to_rv.REG_DATA_BASE: data_base, isa_to_rv.REG_FENCE: 0}.get(off)
            if val is None:
                raise AssertionError(f"load from 0x{addr:x}")
        elif op == 0x23 and f3 == 2:
            imm = _sx(((w >> 25) << 5) | ((w >> 7) & 31), 12)
            off = ((rs1 + imm) & 0xFFFFFFFF) - isa_to_rv.MMIO_BASE
            if off in regs:
                regs[off] = rs2 & 0xFFFFFFFF
            elif off == isa_to_rv.REG_DOORBELL:
                m, n, k = regs[0] & 0xFFFF, regs[0] >> 16, regs[4] & 0xFFFF
                if not (1 <= m <= limits.rows and 1 <= n <= limits.columns and 1 <= k <= limits.max_k):
                    return jobs, isa.ERR_BAD_SHAPE
                jobs.append((m, n, k, regs[8], regs[12], regs[16]))
            elif off == isa_to_rv.REG_DONE:
                return jobs, rs2 & 0xFF
            else:
                raise AssertionError(f"store to MMIO offset 0x{off:x}")
        elif op == 0x63 and f3 == 1:
            imm = _sx((((w >> 31) & 1) << 12) | (((w >> 7) & 1) << 11) | (((w >> 25) & 0x3F) << 5) |
                      (((w >> 8) & 0xF) << 1), 13)
            if rs1 != rs2:
                nxt = pc + imm
        elif op == 0x6F:
            imm = _sx((((w >> 31) & 1) << 20) | (((w >> 12) & 0xFF) << 12) | (((w >> 20) & 1) << 11) |
                      (((w >> 21) & 0x3FF) << 1), 21)
            val, nxt = pc + 4, pc + imm
        else:
            raise AssertionError(f"unexpected instruction 0x{w:08x} at 0x{pc:x}")
        if val is not None and rd:
            x[rd] = val & 0xFFFFFFFF
        pc = nxt
    raise AssertionError("program did not end")


class IsaToRvTest(unittest.TestCase):
    def test_translated_programs_issue_the_same_jobs(self):
        for isa_case, rv_case, size in isa_to_rv.build():
            with self.subTest(case=isa_case.name):
                words = isa_case.image[:isa_case.prog_len]
                if isa_case.name == "illegal_opcode":
                    words = words[:-2] + [isa.end()]
                memory = np.zeros(1 << 16, np.uint8)
                want_error = 0
                try:
                    trace = isa_sim.run(words, memory, isa.Limits(16, 16, 256), log_jobs=True)
                    want_jobs = trace.jobs_log
                except isa_sim.ISAFault as fault:
                    want_error, want_jobs = fault.code, []
                if isa_case.name == "illegal_opcode":
                    want_error = isa.ERR_ILLEGAL_OPCODE
                code = b"".join(w.to_bytes(8, "little") for w in rv_case.image[:rv_case.prog_len])
                jobs, end = run_rv(code, rv_case.data_base)
                self.assertEqual(end, want_error)
                self.assertEqual(end, isa_case.error)
                self.assertEqual(len(jobs), isa_case.jobs)
                base = rv_case.data_base
                self.assertEqual(jobs, [(m, n, k, a + base, b + base, c + base)
                                        for m, n, k, a, b, c in want_jobs])
                # Data region moved intact, program bytes counted.
                self.assertEqual(rv_case.image[base // 8:], isa_case.image[isa_case.data_base // 8:])
                self.assertEqual(rv_case.prog_len, (size + 7) // 8)
