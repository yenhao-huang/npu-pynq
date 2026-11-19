"""delegate/host.py: a package's delegates run as INT8 GEMMs on this CPU.

Each case writes the two files the host backend reads, a device's program
table and weights.bin holding every program's B as pack_b lays it out, then
calls the backend through its C entry point, as __delegate__ would, and
compares the result with numpy's A @ B.
"""

import ctypes
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.runtime.delegate import DelegateError
from src.runtime.delegate.host import HostBackend, read_programs, unpack_b
from src.vendors.npu_pynq.spec.layout import GemmPlan, pack_b

# (m, n, k, tm, tn, tk): partial row, column and K tiles, a single tile, several K chunks.
SHAPES = [(3, 40, 300, 16, 16, 256), (1, 16, 16, 1, 16, 256), (16, 33, 600, 16, 16, 256)]


def write_package(root: Path, device: str, shapes, segments_per_program: int = 1) -> list[np.ndarray]:
    """Program table and weights.bin for ``shapes``; returns each program's B."""
    rng = np.random.default_rng(0)
    blobs, lines, bs, offset = [], [], [], 0
    for index, shape in enumerate(shapes):
        plan = GemmPlan(*shape)
        b = rng.integers(-128, 128, (plan.k, plan.n), dtype=np.int8)
        packed = pack_b(plan, b)
        bs.append(b)
        blobs.append(packed)
        lines.append(f'task {index} {" ".join(str(v) for v in shape)} 0 0 0 {segments_per_program}')
        # Split the column tiles into consecutive segments, as the back ends do.
        cuts = np.linspace(0, plan.nt, segments_per_program + 1).astype(int)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            start, size = lo * plan.b_panel, (hi - lo) * plan.b_panel
            if device == 'npu':
                lines.append(f'seg 0 0 {offset + start} {size} 0')
            else:
                lines.append(f'seg 0 0 0 {offset + start} {size}')
        offset += packed.size
    header = 'npu-programs 1 16 16 256 0' if device == 'npu' else 'rv-programs 1 16 16 256 0'
    table = 'programs.txt' if device == 'npu' else 'rv_programs.txt'
    (root / table).write_text('\n'.join([f'{header} {len(shapes)}', *lines]) + '\n')
    (root / 'weights.bin').write_bytes(np.concatenate(blobs).tobytes())
    return bs


def call(backend: HostBackend, index: int, a: np.ndarray, n: int) -> tuple[int, np.ndarray]:
    a = np.ascontiguousarray(a, np.int8)
    out = np.zeros((a.shape[0], n), np.int32)
    buffers = (ctypes.c_void_p * 2)(a.ctypes.data, out.ctypes.data)
    fn = ctypes.cast(backend.execute, type(backend._fn))
    return fn(index, buffers, 2), out


class HostBackendTests(unittest.TestCase):
    def check(self, device: str, segments: int) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bs = write_package(root, device, SHAPES, segments)
            backend = HostBackend(root, device)
            rng = np.random.default_rng(1)
            for index, (shape, b) in enumerate(zip(SHAPES, bs)):
                m, n, k = shape[:3]
                a = rng.integers(-128, 128, (m, k), dtype=np.int8)
                rc, out = call(backend, index, a, n)
                self.assertEqual(rc, 0, backend.message)
                np.testing.assert_array_equal(out, a.astype(np.int32) @ b.astype(np.int32))
            stats = backend.stats()
            self.assertEqual(stats['calls'], len(SHAPES))
            self.assertEqual(stats['jobs'], 0)
            self.assertEqual(stats['macs'], sum(m * n * k for m, n, k, *_ in SHAPES))
            backend.close()

    def test_npu_package(self):
        self.check('npu', 1)

    def test_risc_v_npu_package(self):
        self.check('risc_v_npu', 1)

    def test_segments_join_into_one_b(self):
        self.check('risc_v_npu', 2)
        self.check('npu', 2)

    def test_unpack_inverts_pack_b(self):
        plan = GemmPlan(5, 37, 513, 16, 16, 256)
        b = np.random.default_rng(2).integers(-128, 128, (plan.k, plan.n), dtype=np.int8)
        np.testing.assert_array_equal(unpack_b(plan, pack_b(plan, b)), b)

    def test_a_bad_index_reports_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_package(root, 'npu', SHAPES[:1])
            backend = HostBackend(root, 'npu')
            rc, _ = call(backend, 0, np.zeros((3, 300), np.int8), 40)
            self.assertEqual(rc, 0)
            buffers = (ctypes.c_void_p * 2)()
            fn = ctypes.cast(backend.execute, type(backend._fn))
            self.assertEqual(fn(5, buffers, 2), 1)
            self.assertEqual(backend.stats()['error_program'], 5)
            self.assertIn('IndexError', backend.message)
            backend.close()

    def test_unknown_device_and_missing_table_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DelegateError):
                read_programs(Path(tmp), 'gpu')
            with self.assertRaises(DelegateError):
                read_programs(Path(tmp), 'npu')


if __name__ == '__main__':
    unittest.main()
