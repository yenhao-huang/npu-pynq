"""Tests for the RV32I assembler, the reference ISS and the scalar-core vectors."""

import unittest
from pathlib import Path

from src.test.model.rv32_asm import VFENCE_WORD, AsmError, assemble, encode_vector
from src.test.model.rv32_iss import (
    FAULT_DBUS,
    FAULT_FETCH_MISALIGNED,
    FAULT_IBUS,
    FAULT_ILLEGAL,
    FAULT_LOAD_MISALIGNED,
    FAULT_STORE_MISALIGNED,
    HALT_EBREAK,
    HALT_ECALL,
    RV32ISS,
)
from src.test.vectors.generate_scalar_vectors import OUTPUT, build_cases, render

FIXTURE = Path("src/test/vectors/scalar_core_programs.txt")


def words(source, base=0):
    return assemble(source, base).words


def run(source, start=0, mem_size=0x10000):
    prog = assemble(source)
    iss = RV32ISS(mem_size)
    iss.load_words(prog.image())
    return iss, iss.run(start), prog


class EncodingTests(unittest.TestCase):
    # Reference encodings (GNU as / the RISC-V specification).
    KNOWN = [
        ("addi x1, x0, 1", 0x00100093),
        ("add x3, x1, x2", 0x002081B3),
        ("sub x3, x1, x2", 0x402081B3),
        ("lui x5, 0x12345", 0x123452B7),
        ("auipc a0, 0xfffff", 0xFFFFF517),
        ("lw x5, 8(x2)", 0x00812283),
        ("sw x5, 12(x2)", 0x00512623),
        ("lbu a0, -1(sp)", 0xFFF14503),
        ("sh a1, 6(a0)", 0x00B51323),
        ("srai x1, x1, 3", 0x4030D093),
        ("slli t0, t1, 31", 0x01F31293),
        ("sltiu a0, a1, -1", 0xFFF5B513),
        ("sra a0, a1, a2", 0x40C5D533),
        ("and s1, s2, s3", 0x013974B3),
        ("jalr ra, 0(t0)", 0x000280E7),
        ("ecall", 0x00000073),
        ("ebreak", 0x00100073),
        ("fence", 0x0FF0000F),
        ("fence.i", 0x0000100F),
        ("nop", 0x00000013),
        ("ret", 0x00008067),
        ("mv a0, a1", 0x00058513),
    ]

    def test_known_encodings(self):
        for text, want in self.KNOWN:
            with self.subTest(text=text):
                self.assertEqual(words(text), [want])

    def test_branch_and_jump_offsets(self):
        prog = assemble("""
        top:
            beq x1, x2, fwd     # +8
            j top               # -4
        fwd:
            bnez a0, top        # -8
            jal fwd             # -4, rd = ra
            bgt a0, a1, top     # blt a1, a0
        """)
        self.assertEqual(prog.words[0], 0x00208463)
        self.assertEqual(prog.words[1], 0xFFDFF06F)
        self.assertEqual(prog.words[2], 0xFE051CE3)
        self.assertEqual(prog.words[3], 0xFFDFF0EF)
        self.assertEqual(prog.words[4], 0xFEA5C8E3)
        self.assertEqual(prog.symbols, {"top": 0, "fwd": 8})

    def test_li_and_la(self):
        self.assertEqual(words("li a0, -2048"), [0x80000513])
        self.assertEqual(words("li t0, 0x80000000"), [0x800002B7, 0x00028293])
        # Low half >= 0x800 borrows from the upper part.
        self.assertEqual(words("li a0, 0x12345fff"), [0x12346537, 0xFFF50513])
        prog = assemble("la a0, data\n.org 0x1000\ndata: .word 7")
        self.assertEqual(prog.words[:2], [0x00001537, 0x00050513])
        self.assertEqual(prog.words[-1], 7)
        self.assertNotIn(8, prog.image())       # .org fill is not part of the image

    def test_expressions_and_directives(self):
        prog = assemble(".equ N, 3\n.word N+4, -1, . , 0x10 - N\nend:")
        self.assertEqual(prog.words, [7, 0xFFFFFFFF, 8, 13])
        self.assertEqual(prog.symbols["end"], 16)
        self.assertEqual(assemble("nop", base=0x400).image(), {0x400: 0x13})

    def test_vector_encoding_matches_architecture_table(self):
        word = words("vadd.i16 v3, v1, v6")[0]
        self.assertEqual(word & 0x7F, 0b0001011)
        self.assertEqual((word >> 25) & 0x1F, 3)                    # op
        self.assertEqual((word >> 12) & 3, 1)                       # type
        self.assertEqual((word >> 7) & 7, 3)                        # vd
        self.assertEqual((word >> 20) & 7, 1)                       # vs1
        self.assertEqual(((word >> 30) << 1) | ((word >> 14) & 1), 6)   # vs2
        self.assertEqual((word >> 15) & 0x1F, 0)                    # rs1 = x0
        self.assertEqual(words("vld v1, x5"), [encode_vector(0, 0, 1, 0, 0, 5)])
        self.assertEqual(words("vst v2, x6"), [encode_vector(1, 0, 0, 2, 0, 6)])
        self.assertEqual(words("vsplat.i16 v0, x7"), [encode_vector(2, 1, 0, 0, 0, 7)])
        self.assertEqual(words("vsra.i32 v1, v1, x8"), [encode_vector(9, 2, 1, 1, 0, 8)])
        self.assertEqual(words("vrelu.f16 v4, v5"), [encode_vector(8, 3, 4, 5, 0, 0)])
        self.assertEqual(words("vmov v7, v2"), [encode_vector(10, 0, 7, 2, 0, 0)])
        self.assertEqual(words("vcmd 17, 2, v1, v2, v7, a0"), [encode_vector(17, 2, 1, 2, 7, 10)])
        self.assertEqual(words("vfence"), [VFENCE_WORD])
        self.assertEqual(VFENCE_WORD, 0x3E00000B)

    def test_errors(self):
        for text in ("addi x1, x0, 2048", "add x1, x2", "lw x1, 0(x32)", "beq x0, x0, nowhere",
                     "slli x1, x1, 32", "vadd v1, v2, v3", "vadd.i64 v1, v2, v3", "vld v8, x1",
                     "frob x1", "dup:\ndup:", ".org 4\n.org 0", "beq x0, x0, 3"):
            with self.subTest(text=text):
                with self.assertRaises(AsmError):
                    assemble(text)


class IssTests(unittest.TestCase):
    def test_alu_and_halt(self):
        iss, res, _ = run("""
            li a0, -7
            srai a1, a0, 1
            srli a2, a0, 28
            sltu a3, a0, x0
            slt a4, a0, x0
            lui a5, 0xfffff
            auipc a6, 0
            ebreak
        """)
        self.assertEqual(res.status, "halt")
        self.assertEqual(res.cause, HALT_EBREAK)
        self.assertEqual(res.instret, 8)
        self.assertEqual(iss.regs[10:17], [0xFFFFFFF9, 0xFFFFFFFC, 0xF, 0, 1, 0xFFFFF000, 24])

    def test_loads_stores_and_x0(self):
        iss, res, _ = run("""
            li s0, 0x100
            li t0, 0x80ff7f01
            sw t0, 0(s0)
            lb a0, 3(s0)
            lbu a1, 3(s0)
            lh a2, 2(s0)
            lhu a3, 0(s0)
            sb t0, 5(s0)
            sh t0, 10(s0)
            addi x0, x0, 1
            ecall
        """)
        self.assertEqual(res.cause, HALT_ECALL)
        self.assertEqual(iss.regs[10:14], [0xFFFFFF80, 0x80, 0xFFFF80FF, 0x7F01])
        self.assertEqual(iss.word(0x104), 0x00000100)
        self.assertEqual(iss.word(0x108), 0x7F010000)
        self.assertEqual(iss.regs[0], 0)
        self.assertEqual(iss.written, {0x100, 0x104, 0x108})

    def test_faults_have_no_side_effects(self):
        cases = [
            ("li t1, 5\nli t0, 0x102\nlw t1, 0(t0)", FAULT_LOAD_MISALIGNED, 8),
            ("li t1, 5\nli t0, 0x101\nsh t1, 0(t0)", FAULT_STORE_MISALIGNED, 8),
            ("li t1, 5\njal t1, . + 6", FAULT_FETCH_MISALIGNED, 4),
            ("li t1, 5\n.word 0x30200073", FAULT_ILLEGAL, 4),
            ("li t1, 5\nli t0, 0x20000\nlw t1, 0(t0)", FAULT_DBUS, 12),
        ]
        for source, cause, pc in cases:
            with self.subTest(source=source):
                iss, res, _ = run(source)
                self.assertEqual((res.status, res.cause, res.stop_pc), ("fault", cause, pc))
                self.assertEqual(iss.regs[6], 5)
                self.assertEqual(iss.written, set())
        iss, res, _ = run("li t0, 0x20000\njr t0")
        self.assertEqual((res.cause, res.stop_pc, res.instret), (FAULT_IBUS, 0x20000, 3))
        iss, res, _ = run("nop", start=2)
        self.assertEqual((res.cause, res.stop_pc, res.instret), (FAULT_FETCH_MISALIGNED, 2, 0))
        # A start PC outside memory: the first fetch is a bus error.
        iss, res, _ = run("nop", start=0x40000)
        self.assertEqual((res.cause, res.stop_pc, res.instret), (FAULT_IBUS, 0x40000, 0))

    def test_vector_log_and_restart(self):
        iss, res, prog = run("""
            li a0, 9
            vsplat.i8 v2, a0
            vfence
            vadd.i32 v1, v2, v3
            ecall
        second:
            addi a0, a0, 1
            vst v1, a0
            ecall
        """)
        self.assertEqual([c.as_tuple() for c in iss.vcmds],
                         [(2, 0, 2, 0, 0, 9), (3, 2, 1, 2, 3, 0)])
        self.assertEqual(iss.fences, [1])
        res2 = iss.run(prog.symbols["second"])
        self.assertEqual((res2.status, res2.instret), ("halt", 3))
        self.assertEqual(iss.vcmds[-1].as_tuple(), (1, 0, 0, 1, 0, 10))

    def test_cycle_model(self):
        def cycles(source, latency):
            prog = assemble(source)
            iss = RV32ISS(0x10000, latency=latency)
            iss.load_words(prog.image())
            return iss.run(0).cycles

        # start -> first DECODE (1 + L), per instruction DECODE -> DECODE, and
        # the stop (ECALL: its DECODE cycle).
        src = "add a0, a0, a0\nlw a1, 0(x0)\nvfence\nvadd.i8 v1, v2, v3\necall"
        self.assertEqual(cycles(src, 1), 2 + 3 + 5 + 5 + 3 + 1)
        self.assertEqual(cycles(src, 0), 1 + 3 + 4 + 4 + 3 + 1)
        self.assertEqual(cycles(src, 3), 4 + 5 + 7 + 7 + 5 + 1)
        # The branch decision is made in DECODE: taken or not, a branch fetches
        # its successor from EXEC like JAL.
        src = "j l1\nl1: beq x0, x0, l2\nl2: bne x0, x0, l2\necall"
        self.assertEqual(cycles(src, 1), 2 + 3 + 3 + 3 + 1)
        # Stops: misaligned access in EXEC (2), data bus error (3 + L),
        # instruction bus error where the next DECODE would be (0).
        self.assertEqual(cycles("li t0, 2\nlw t1, 0(t0)", 1), 2 + 3 + 2)
        self.assertEqual(cycles("lui t0, 0x100\nlw t1, 0(t0)", 1), 2 + 3 + 4)
        self.assertEqual(cycles("lui t0, 0x100\nlw t1, 0(t0)", 0), 1 + 3 + 3)
        self.assertEqual(cycles("lui t0, 0x30\njr t0", 1), 2 + 3 + 3)


class VectorFileTests(unittest.TestCase):
    def test_fixture_is_reproducible(self):
        self.assertEqual(OUTPUT.resolve(), FIXTURE.resolve())
        self.assertEqual(render(build_cases()), FIXTURE.read_text(encoding="ascii"))

    def test_programs_reach_their_end(self):
        cases = {c["name"]: c for c in build_cases()}
        fib = dict(cases["fibonacci"]["mem"])
        self.assertEqual(fib[0x8000], 55)
        values = [dict(cases["bubble_sort"]["mem"])[0x8000 + 4 * i] for i in range(16)]
        signed = [v - (1 << 32) if v >> 31 else v for v in values]
        self.assertEqual(signed, sorted(signed))
        for name in ("coverage", "memcpy", "bubble_sort", "fibonacci", "vector", "random0"):
            self.assertEqual(cases[name]["runs"][-1][1].status, "halt", name)
        causes = {res.cause for _, res in cases["faults"]["runs"] if res.status == "fault"}
        self.assertEqual(causes, {1, 2, 3, 4, 5, 6})
        # Last memory word: a vector command retires (and is sent) before the
        # bus error of its successor fetch; a load's own data bus error wins.
        for name in ("edge_vcmd", "edge_vcmd_ideal"):
            runs = cases[name]["runs"]
            self.assertTrue(all((r.cause, r.stop_pc) == (FAULT_IBUS, 0x10000) for _, r in runs))
            self.assertEqual(len(cases[name]["vcmds"]), len(runs))
        for name in ("edge_dbus", "edge_dbus_ideal", "edge_dbus_slow3", "edge_dbus_slow6"):
            self.assertTrue(all((r.cause, r.stop_pc) == (FAULT_DBUS, 0xFFFC)
                                for _, r in cases[name]["runs"]))


if __name__ == "__main__":
    unittest.main()
