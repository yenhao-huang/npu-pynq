"""libnpu_rt on the host simulator: a delegate's output is the subgraph's
result, the M x N product, with the NPU's K chunks summed by the library."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

import numpy as np

from src.vendors.npu_pynq.spec import isa, layout

ADDR_B = 0x12   # opcode of the words the loader relocates to the stage window


def write_package(root: Path, m: int, n: int, k: int, tk: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One NPU program, C[m x n] = A[m x k] @ B[k x n] in K chunks of ``tk``, in libnpu_rt's package format."""
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, (m, k), dtype=np.int8)
    b = rng.integers(-128, 128, (k, n), dtype=np.int8)
    limits = isa.Limits()
    plan = layout.plan_gemm(m, n, k, limits, tk)
    c_base = isa.align(plan.a_bytes, 64)
    words = layout.encode(plan, 0, 0, c_base) + [isa.end()]
    relocations = [i for i, w in enumerate(words) if w >> 56 == ADDR_B]
    b_bytes = plan.b_bytes
    (root / 'programs.bin').write_bytes(struct.pack(f'<{len(words)}Q', *words))
    (root / 'programs.txt').write_text(
        f'npu-programs 1 {limits.rows} {limits.columns} {limits.max_k} {len(words)} 1\n'
        f'task 0 {m} {n} {k} {plan.tm} {plan.tn} {plan.tk} {plan.a_bytes} {c_base} {plan.c_bytes} 1\n'
        f'seg 0 {len(words)} 0 {b_bytes} {len(relocations)} {" ".join(map(str, relocations))}\n')
    (root / 'weights.bin').write_bytes(layout.pack_b(plan, b).tobytes())
    return a, b


@unittest.skipUnless(sys.platform.startswith('linux') and (shutil.which('c++') or shutil.which('cmake')),
                     'libnpu_rt builds on Linux with a C++ compiler')
class KChunkSumTest(unittest.TestCase):
    def test_the_output_is_the_full_product(self) -> None:
        from src.runtime.delegate.npu import NpuBackend

        # (m, n, k, tk): one chunk, several chunks with a short last one, and
        # several row and column tiles with two chunks.
        for m, n, k, tk in ((4, 4, 8, 8), (5, 7, 40, 8), (20, 18, 300, 256)):
            with self.subTest(m=m, n=n, k=k, tk=tk), tempfile.TemporaryDirectory() as tmp:
                a, b = write_package(Path(tmp), m, n, k, tk)
                backend = NpuBackend(Path(tmp))
                lib = backend.lib
                lib.npu_rt_execute.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
                # One guard row past M x N: the library writes M x N, not Kt planes.
                c = np.full((m + 1, n), 0x5A5A5A5A, np.int32)
                buffers = (ctypes.c_void_p * 2)(a.ctypes.data, c.ctypes.data)
                try:
                    for _ in range(2):   # a second call overwrites, it does not accumulate
                        self.assertEqual(lib.npu_rt_execute(0, buffers, 2), 0, lib.npu_rt_error())
                        np.testing.assert_array_equal(c[:m], a.astype(np.int32) @ b.astype(np.int32))
                    self.assertTrue((c[m] == 0x5A5A5A5A).all())
                finally:
                    backend.close()


if __name__ == '__main__':
    unittest.main()
