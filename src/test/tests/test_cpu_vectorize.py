"""NEON vectorization of the CPU code (pynqo-backend, src/export/backend/cpu/toolchain.cpp).

The back end moves each linalg op's unit-stride dimension innermost, then
asks LLVM to vectorize every innermost loop. The interchange keeps every
output's summation order, so results stay bit-identical; the board library
uses NEON q registers. Hand-written linalg, so only pynqo-backend is needed;
running the library also needs Linux."""

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

from src.export import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_export_middle_end import tool_is_built  # noqa: E402

TOOL = tool_is_built(pipeline.backend_tool())
LLVM_BIN = '/usr/lib/llvm-22/bin'
B, M, N, D = 2, 16, 32, 8          # attention scores: [B, M, D] x [B, D, N]
K, OUT = 24, 40                    # INT8 linear: [M, K] x [K, OUT]

WEIGHT = np.random.default_rng(7).integers(-128, 128, size=(K, OUT), dtype=np.int8)


def write_model(d: Path) -> Path:
    """Q·Kᵀ as attention lowers it (K stored [D, N], so the innermost
    reduction over D reads K with a stride) and a named INT8 matmul (B stored
    [K, OUT], same stride problem)."""
    rows = ', '.join('[' + ', '.join(str(int(v)) for v in row) + ']' for row in WEIGHT)
    q, k, s = f'tensor<{B}x{M}x{D}xf32>', f'tensor<{B}x{D}x{N}xf32>', f'tensor<{B}x{M}x{N}xf32>'
    r = f'tensor<{B}x{M}xf32>'
    x, w, y = f'tensor<{M}x{K}xi8>', f'tensor<{K}x{OUT}xi8>', f'tensor<{M}x{OUT}xi32>'
    model = d / 'model.mlir'
    model.write_text(f'''#q = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#k = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#s = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#a = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#r = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {{pynqo.devices = ["cpu"]}} {{
  func.func @scores(%q: {q}, %k: {k}) -> {s} {{
    %c0 = arith.constant 0.0 : f32
    %e = tensor.empty() : {s}
    %z = linalg.fill ins(%c0 : f32) outs(%e : {s}) -> {s}
    %r = linalg.generic {{indexing_maps = [#q, #k, #s],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction"]}}
        ins(%q, %k : {q}, {k}) outs(%z : {s}) attrs = {{pynqo.kernel_id = 1 : i32}} {{
    ^bb0(%a: f32, %b: f32, %acc: f32):
      %m = arith.mulf %a, %b : f32
      %t = arith.addf %acc, %m : f32
      linalg.yield %t : f32
    }} -> {s}
    return %r : {s}
  }}
  func.func @rowsum(%a: {q}) -> {r} {{
    %c0 = arith.constant 0.0 : f32
    %e = tensor.empty() : {r}
    %z = linalg.fill ins(%c0 : f32) outs(%e : {r}) -> {r}
    %r = linalg.generic {{indexing_maps = [#a, #r], iterator_types = ["parallel", "parallel", "reduction"]}}
        ins(%a : {q}) outs(%z : {r}) {{
    ^bb0(%v: f32, %acc: f32):
      %t = arith.addf %acc, %v : f32
      linalg.yield %t : f32
    }} -> {r}
    return %r : {r}
  }}
  func.func @linear(%x: {x}) -> {y} {{
    %w = arith.constant dense<[{rows}]> : {w}
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : {y}
    %z = linalg.fill ins(%c0 : i32) outs(%e : {y}) -> {y}
    %r = linalg.matmul {{pynqo.kernel_id = 2 : i32}} ins(%x, %w : {x}, {w}) outs(%z : {y}) -> {y}
    return %r : {y}
  }}
}}
''', encoding='utf-8')
    return model


def backend(model: Path, out: Path, cpu: str, extra: str = '') -> tuple[dict, Path]:
    pipeline._linux(f"{pipeline.backend_tool()} '{pipeline._linux_path(model)}' "
                    f"--out '{pipeline._linux_path(out)}' --devices cpu --cpus {cpu} {extra}")
    compiled = json.loads((out / 'compile.json').read_text())
    (lib,) = compiled['libraries']
    return compiled, out / lib


def functions(listing: str) -> dict[str, str]:
    """llvm-objdump -d output, split by function."""
    parts = re.split(r'^[0-9a-f]+ <([^>]+)>:$', listing, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def sequential_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Q·Kᵀ in float32, each output summed over D in order: what scalar code
    computes, rounding after every multiply and add."""
    acc = np.zeros((B, M, N), np.float32)
    for d in range(D):
        acc = acc + q[:, :, d, None] * k[:, None, d, :]
    return acc


@unittest.skipUnless(TOOL, 'pynqo-backend is required')
class NeonCodeTest(unittest.TestCase):
    def test_board_library_uses_neon_q_registers(self):
        with tempfile.TemporaryDirectory() as d:
            _, lib = backend(write_model(Path(d)), Path(d) / 'pkg', 'arm-cortex-a9', '--profile-kernels')
            listing = Path(d) / 'model.s'
            pipeline._linux(f"{LLVM_BIN}/llvm-objdump -d --mattr=+neon '{pipeline._linux_path(lib / 'model.so')}' "
                            f"> '{pipeline._linux_path(listing)}'")
            code = functions(listing.read_text())
            table = json.loads((lib / 'kernels.json').read_text())['kernels']
        # memrefCopy (C, compiled by clang) has NEON code of its own: look in
        # the kernels' functions only.
        self.assertTrue(re.search(r'\bv(mul|mla|add)\.f32\s+q\d+', code['scores']),
                        'Q·Kᵀ has no NEON float arithmetic')
        self.assertTrue(re.search(r'\bv(mul|mla|add)\.i32\s+q\d+', code['linear']),
                        'the INT8 matmul has no NEON integer arithmetic')
        # The generalized matmul keeps its kernel id for the profiler.
        self.assertEqual(sorted(k['id'] for k in table if k['kind'] == 'kernel'), [1, 2])


@unittest.skipUnless(TOOL and sys.platform.startswith('linux'), 'pynqo-backend and Linux are required')
class InterchangeIsExactTest(unittest.TestCase):
    def test_results_are_bit_identical_to_in_order_sums(self):
        from src.runtime.share.loader import ExportedPackage

        rng = np.random.default_rng(3)
        q = rng.standard_normal((B, M, D)).astype(np.float32)
        k = rng.standard_normal((B, D, N)).astype(np.float32)
        x = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'pkg'
            compiled, _ = backend(write_model(Path(d)), out, 'native')
            (out / 'manifest.json').write_text(json.dumps({**compiled, 'format': pipeline.PACKAGE_FORMAT}))
            package = ExportedPackage(out)
            scores = np.zeros((B, M, N), np.float32)
            package.call('scores', [q, k], [scores])
            linear = np.zeros((M, OUT), np.int32)
            package.call('linear', [x], [linear])
            rowsum = np.zeros((B, M), np.float32)
            package.call('rowsum', [q], [rowsum])
        np.testing.assert_array_equal(scores, sequential_scores(q, k))
        # A float reduction left innermost is not reordered (vectorizeHints
        # gives its loop no hint).
        expected = np.zeros((B, M), np.float32)
        for d in range(D):
            expected = expected + q[:, :, d]
        np.testing.assert_array_equal(rowsum, expected)
        np.testing.assert_array_equal(linear, x.astype(np.int32) @ WEIGHT.astype(np.int32))


if __name__ == '__main__':
    unittest.main()
