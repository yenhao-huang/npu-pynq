import unittest

import numpy as np

from src.isa import isa, layout, sim


def run_gemm(m, n, k, limits=isa.Limits(), tk=None, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, (m, k), dtype=np.int8)
    b = rng.integers(-128, 128, (k, n), dtype=np.int8)
    plan = layout.plan_gemm(m, n, k, limits, tk)
    a_base = 0
    b_base = isa.align(plan.a_bytes)
    c_base = b_base + isa.align(plan.b_bytes)
    memory = np.zeros(c_base + plan.c_bytes, np.uint8)
    memory[a_base:a_base + plan.a_bytes] = layout.pack_a(plan, a).view(np.uint8)
    memory[b_base:b_base + plan.b_bytes] = layout.pack_b(plan, b).view(np.uint8)
    program = layout.encode(plan, a_base, b_base, c_base) + [isa.end()]
    trace = sim.run(program, memory, limits)
    partial = layout.unpack_c(plan, memory[c_base:])
    return a, b, plan, trace, partial


class EncodingTest(unittest.TestCase):
    def test_round_trip(self):
        words = [isa.shape(16, 9, 200), isa.addr_a(8), isa.incr(16, 4096, 1024), isa.repeat(7), isa.gemm(6), isa.end()]
        self.assertEqual(isa.unpack(isa.pack(words)), words)
        f = isa.decode(words[0]).fields()
        self.assertEqual((f["m"], f["n"], f["k"]), (16, 9, 200))
        self.assertEqual(isa.decode(words[2]).fields(), {"a_inc": 16, "b_inc": 4096, "c_inc": 1024})
        self.assertIn("repeat  count=7", isa.disassemble(words))

    def test_rejects_unaligned_and_oversized_fields(self):
        with self.assertRaises(isa.ISAError):
            isa.addr_a(3)
        with self.assertRaises(isa.ISAError):
            isa.shape(256, 1, 1)
        with self.assertRaises(isa.ISAError):
            isa.incr(a_inc=4)


class SimulatorTest(unittest.TestCase):
    def test_tiled_gemm_matches_numpy(self):
        for m, n, k, tk in [(1, 1, 1, None), (16, 16, 256, None), (5, 37, 300, None),
                            (33, 48, 576, 128), (1, 49, 1536, None), (17, 3, 9, None)]:
            with self.subTest(m=m, n=n, k=k, tk=tk):
                a, b, plan, trace, partial = run_gemm(m, n, k, tk=tk)
                np.testing.assert_array_equal(partial.sum(0), a.astype(np.int32) @ b.astype(np.int32))
                self.assertEqual(trace.jobs, plan.jobs)
                self.assertEqual(trace.macs, m * n * k)

    def test_small_array_limits(self):
        limits = isa.Limits(rows=2, columns=2, max_k=8)
        a, b, plan, trace, partial = run_gemm(5, 7, 20, limits)
        np.testing.assert_array_equal(partial.sum(0), a.astype(np.int32) @ b.astype(np.int32))
        self.assertEqual(plan.jobs, 3 * 4 * 3)

    def test_faults(self):
        memory = np.zeros(4096, np.uint8)
        with self.assertRaises(sim.ISAFault) as bad_shape:
            sim.run([isa.shape(17, 1, 1), isa.gemm(), isa.end()], memory)
        self.assertEqual(bad_shape.exception.code, isa.ERR_BAD_SHAPE)
        with self.assertRaises(sim.ISAFault) as illegal:
            sim.run([0x7F << 56, isa.end()], memory)
        self.assertEqual(illegal.exception.code, isa.ERR_ILLEGAL_OPCODE)
        with self.assertRaises(sim.ISAFault):
            sim.run([isa.nop()], memory)


if __name__ == "__main__":
    unittest.main()


class HardwareContractTest(unittest.TestCase):
    def test_vectors_are_current(self):
        from src.test.vectors import generate_isa_vectors as gen
        self.assertEqual(gen.OUTPUT.read_text(), gen.render(), "regenerate isa_frontend_16x16.txt")

    def test_rtl_opcodes_and_registers_match_the_spec(self):
        import re
        from pathlib import Path
        from src.test.model.abi import Capability, IsaRegister
        root = Path(__file__).resolve().parents[3] / "src/hw/rtl/npu_matrix/npu_accelerator"
        decode = (root / "npu_isa_frontend/npu_isa_decode.sv").read_text()
        rtl_ops = {name.lower(): int(value, 16)
                   for name, value in re.findall(r"OP_(\w+) = 8'h([0-9a-fA-F]+)", decode)}
        self.assertEqual(rtl_ops, isa.OPCODES)
        regs = (root / "dma_axi/npu_axi_lite_regs.sv").read_text()
        rtl_regs = {name: int(value, 16)
                    for name, value in re.findall(r"REG_(ISA_\w+|PROG_\w+|DATA_BASE) = 8'h([0-9a-fA-F]+)", regs)}
        expected = {("ISA_" + r.name if r.name in ("CONTROL", "STATUS", "ERROR", "PC", "JOBS", "CYCLES",
                                                   "INSTRUCTIONS", "VERSION") else r.name): r.value
                    for r in IsaRegister}
        self.assertEqual(rtl_regs, expected)
        self.assertIn("32'h0000007b", regs)
        self.assertTrue(0x7B & Capability.ISA_FRONTEND)
