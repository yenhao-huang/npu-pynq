"""MLIR/LLVM compiler: partition, ISA encoding parity, and host execution.

Skipped when no LLVM 22 with MLIR is installed (the hosted CI image).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.compiler import toolchain
from src.isa import isa, layout

try:
    toolchain.llvm_bin()
    HAVE_LLVM = True
except toolchain.ToolchainError:
    HAVE_LLVM = False


def build_mlp(m, k, h, n, bits2, group2, rng):
    from src.compiler.export import ModelGraph
    from src.compiler.mlir_builder import ModuleBuilder, memref_type
    from src.compiler.ops import core, linear as L
    from src.compiler.ops.context import Ctx

    w1 = L.quantize_weight(rng.normal(size=(h, k)).astype(np.float32) * 0.05, 8)
    w2 = L.quantize_weight(rng.normal(size=(n, h)).astype(np.float32) * 0.1, bits2, group=group2)
    bias = np.linspace(-1, 1, h).astype(np.float32)
    mod = ModuleBuilder()
    f = mod.func("forward", [("ws", "memref<?xi8>"), ("x", memref_type([m, k], "f32")),
                             ("out", memref_type([m, n], "f32"))])
    graph = ModelGraph("mlp", mod)
    ctx = Ctx(f, graph, f.arg("ws"))
    x = f.to_tensor(f.arg("x"))
    y = L.linear(ctx, core.relu(f, L.linear(ctx, x, "fc1", w1, bias)), "fc2", w2)
    f.store(y, f.arg("out"))
    f.emit("return")

    def reference(x_np):
        return L.ref_linear(np.maximum(L.ref_linear(x_np, w1, bias), 0), w2)

    return graph, reference


@unittest.skipUnless(HAVE_LLVM, "LLVM 22 with MLIR is not installed")
class CompilerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def compile_and_run(self, bits2, group2, targets=("host",)):
        from src.compiler.export import export
        from src.runtime.compiled import CompiledModel

        rng = np.random.default_rng(bits2)
        m, k, h, n = 5, 300, 48, 40
        graph, reference = build_mlp(m, k, h, n, bits2, group2, rng)
        pkg = export(graph, self.root / f"mlp{bits2}", targets=targets)
        model = CompiledModel(pkg, "sim")
        x = rng.normal(size=(m, k)).astype(np.float32)
        out = np.zeros((m, n), np.float32)
        model.call("forward", "weights", x, out)
        return pkg, out, reference(x), model.stats()

    def test_w8_and_w4_mlp_match_numpy_on_the_simulated_npu(self):
        for bits, group in ((8, 0), (4, 16)):
            with self.subTest(bits=bits):
                pkg, out, ref, stats = self.compile_and_run(bits, group)
                np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
                self.assertEqual(stats["calls"], 2)
                report = json.loads((pkg / "partition.json").read_text())
                self.assertEqual([t["weight"] for t in report["npu_tasks"]], ["fc1.q", "fc2.q"])
                self.assertEqual(report["npu_tasks"][1]["source"], "grouped" if bits == 4 else "linalg.matmul")

    def test_streamed_segments_and_packed_int4(self):
        from src.compiler.export import export
        from src.runtime.compiled import CompiledModel

        rng = np.random.default_rng(11)
        m, k, h, n = 3, 300, 64, 200
        graph, reference = build_mlp(m, k, h, n, 4, 32, rng)
        pkg = export(graph, self.root / "streamed", targets=("host",), segment_bytes=2048)
        table = json.loads((pkg / "programs.json").read_text())
        self.assertGreater(sum(len(t["segments"]) for t in table["tasks"]), 4)
        manifest = json.loads((pkg / "manifest.json").read_text())
        self.assertTrue(manifest["weight_layouts"]["fc2.q"].startswith("npu_b4"))
        x = rng.normal(size=(m, k)).astype(np.float32)
        model = CompiledModel(pkg, "sim")
        self.assertTrue(model.stream_weights, "packed INT4 weights must stream")
        out = np.zeros((m, n), np.float32)
        model.call("forward", "weights", x, out)
        np.testing.assert_allclose(out, reference(x), rtol=1e-5, atol=1e-5)
        # INT8 weights run resident or streamed, with the same result.
        graph, reference = build_mlp(m, k, h, n, 8, 0, rng)
        pkg = export(graph, self.root / "w8", targets=("host",), segment_bytes=2048)
        for stream in (False, True):
            model = CompiledModel(pkg, "sim", stream_weights=stream)
            out = np.zeros((m, n), np.float32)
            model.call("forward", "weights", x, out)
            np.testing.assert_allclose(out, reference(x), rtol=1e-5, atol=1e-5)

    def test_cpp_encoder_matches_python_layout(self):
        pkg, *_ = self.compile_and_run(8, 0)
        table = json.loads((pkg / "programs.json").read_text())
        words = isa.unpack((pkg / "programs.bin").read_bytes())
        for t in table["tasks"]:
            plan = layout.plan_gemm(t["m"], t["n"], t["k"], isa.Limits(), t["tk"])
            self.assertEqual(t["c_base"] + t["c_bytes"], isa.align(plan.a_bytes, 64) + plan.c_bytes)
            for g in t["segments"]:
                expected = layout.encode(plan, 0, 0, isa.align(plan.a_bytes, 64), g["ni_lo"], g["ni_hi"]) + [isa.end()]
                self.assertEqual(words[g["word_offset"]:g["word_offset"] + g["length"]], expected)
                self.assertEqual(g["weight_offset"] - t["weight_offset"], g["ni_lo"] * plan.b_panel)

    def test_board_library_is_armv7_hard_float(self):
        pkg, *_ = self.compile_and_run(8, 0, targets=("host", "pynq"))
        header = (pkg / "pynq/model.so").read_bytes()[:52]
        self.assertEqual(header[:4], b"\x7fELF")
        self.assertEqual(header[4], 1)  # 32-bit
        self.assertEqual(int.from_bytes(header[18:20], "little"), 40)  # EM_ARM
        flags = int.from_bytes(header[36:40], "little")
        self.assertTrue(flags & 0x400, "EF_ARM_ABI_FLOAT_HARD")


if __name__ == "__main__":
    unittest.main()
