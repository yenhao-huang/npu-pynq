from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from runtime.matrix_multiplication import (
    MatrixMultiplicationMetrics,
    MatrixMultiplicationResult,
    TiledMatrixMultiplier,
)


class MutableClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class FakePhysicalRuntime:
    max_m = 2
    max_n = 2
    max_k = 256

    def __init__(self, *, clock: MutableClock | None = None, seconds_per_tile: float = 0.0) -> None:
        self.clock = clock
        self.seconds_per_tile = seconds_per_tile
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        a_matrix: np.ndarray,
        b_matrix: np.ndarray,
        *,
        hardware_timeout_cycles: int,
        software_timeout: float,
    ) -> np.ndarray:
        self.calls.append(
            {
                "a": np.array(a_matrix, copy=True),
                "b": np.array(b_matrix, copy=True),
                "a_contiguous": a_matrix.flags.c_contiguous,
                "b_contiguous": b_matrix.flags.c_contiguous,
                "hardware_timeout_cycles": hardware_timeout_cycles,
                "software_timeout": software_timeout,
            }
        )
        if self.clock is not None:
            self.clock.value += self.seconds_per_tile
        return np.asarray(a_matrix, dtype=np.int32) @ np.asarray(b_matrix, dtype=np.int32)


class MatrixMultiplicationTests(unittest.TestCase):
    def test_public_types_and_single_tile_metrics(self):
        clock = MutableClock()
        runtime = FakePhysicalRuntime(clock=clock, seconds_per_tile=0.25)
        multiplier = TiledMatrixMultiplier(runtime, monotonic=clock)
        a_matrix = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int8)
        b_matrix = np.array([[7, 8], [-9, 10], [11, -12]], dtype=np.int8)

        result = multiplier.run(a_matrix, b_matrix, software_timeout=2.0)

        self.assertIsInstance(result, MatrixMultiplicationResult)
        self.assertIsInstance(result.metrics, MatrixMultiplicationMetrics)
        np.testing.assert_array_equal(
            result.output,
            (a_matrix.astype(np.int64) @ b_matrix.astype(np.int64)).astype(np.int32),
        )
        self.assertEqual(result.output.dtype, np.int32)
        self.assertTrue(result.output.flags.c_contiguous)
        self.assertEqual((result.metrics.m, result.metrics.n, result.metrics.k), (2, 2, 3))
        self.assertEqual(result.metrics.tile_count, 1)
        self.assertEqual(result.metrics.mac_count, 12)
        self.assertEqual(result.metrics.operation_count, 24)
        self.assertEqual(result.metrics.elapsed_seconds, 0.25)
        self.assertEqual(result.metrics.operations_per_second, 96.0)

    def test_edge_tiles_are_dense_and_match_reference(self):
        clock = MutableClock()
        runtime = FakePhysicalRuntime(clock=clock, seconds_per_tile=0.25)
        multiplier = TiledMatrixMultiplier(runtime, monotonic=clock)
        a_storage = np.arange(30, dtype=np.int8).reshape(3, 10)
        b_storage = (np.arange(30, dtype=np.int8).reshape(10, 3) - 8)
        a_matrix = a_storage[:, ::2]
        b_matrix = b_storage[::2, :]
        self.assertFalse(a_matrix.flags.c_contiguous)
        self.assertFalse(b_matrix.flags.c_contiguous)

        result = multiplier.run(
            a_matrix,
            b_matrix,
            hardware_timeout_cycles=777,
            software_timeout=5.0,
        )

        np.testing.assert_array_equal(
            result.output,
            (a_matrix.astype(np.int64) @ b_matrix.astype(np.int64)).astype(np.int32),
        )
        self.assertEqual(result.metrics.tile_count, 4)
        self.assertEqual(result.metrics.elapsed_seconds, 1.0)
        self.assertEqual(result.metrics.operation_count, 90)
        self.assertEqual(result.metrics.operations_per_second, 90.0)
        output_shapes = [
            (call["a"].shape[0], call["b"].shape[1])  # type: ignore[union-attr]
            for call in runtime.calls
        ]
        self.assertEqual(output_shapes, [(2, 2), (2, 1), (1, 2), (1, 1)])
        self.assertTrue(all(call["a_contiguous"] for call in runtime.calls))
        self.assertTrue(all(call["b_contiguous"] for call in runtime.calls))
        self.assertTrue(all(call["hardware_timeout_cycles"] == 777 for call in runtime.calls))
        timeouts = [float(call["software_timeout"]) for call in runtime.calls]
        self.assertEqual(timeouts, [5.0, 4.75, 4.5, 4.25])

    def test_repeated_execution_is_independent(self):
        runtime = FakePhysicalRuntime()
        multiplier = TiledMatrixMultiplier(runtime)
        left = np.array([[1, 2], [3, 4]], dtype=np.int8)
        identity = np.eye(2, dtype=np.int8)

        first = multiplier.run(left, identity)
        second = multiplier.run(-left, identity)

        np.testing.assert_array_equal(first.output, left.astype(np.int32))
        np.testing.assert_array_equal(second.output, -left.astype(np.int32))
        self.assertFalse(np.shares_memory(first.output, second.output))
        self.assertEqual(len(runtime.calls), 2)

    def test_invalid_jobs_never_submit(self):
        valid_a = np.ones((2, 3), dtype=np.int8)
        valid_b = np.ones((3, 2), dtype=np.int8)
        invalid_jobs = [
            ([1, 2], valid_b, {}),
            (valid_a.astype(np.int16), valid_b, {}),
            (valid_a.ravel(), valid_b, {}),
            (np.empty((0, 3), dtype=np.int8), valid_b, {}),
            (valid_a, np.ones((4, 2), dtype=np.int8), {}),
            (np.ones((1, 257), dtype=np.int8), np.ones((257, 1), dtype=np.int8), {}),
            (valid_a, valid_b, {"software_timeout": 0.0}),
            (valid_a, valid_b, {"hardware_timeout_cycles": 0}),
        ]
        for a_matrix, b_matrix, kwargs in invalid_jobs:
            with self.subTest(a_type=type(a_matrix), kwargs=kwargs):
                runtime = FakePhysicalRuntime()
                multiplier = TiledMatrixMultiplier(runtime)
                with self.assertRaises((TypeError, ValueError)):
                    multiplier.run(a_matrix, b_matrix, **kwargs)  # type: ignore[arg-type]
                self.assertEqual(runtime.calls, [])

    def test_logical_deadline_stops_before_next_tile(self):
        clock = MutableClock()
        runtime = FakePhysicalRuntime(clock=clock, seconds_per_tile=1.0)
        multiplier = TiledMatrixMultiplier(runtime, monotonic=clock)
        a_matrix = np.ones((3, 2), dtype=np.int8)
        b_matrix = np.ones((2, 3), dtype=np.int8)

        with self.assertRaises(TimeoutError):
            multiplier.run(a_matrix, b_matrix, software_timeout=0.5)

        self.assertEqual(len(runtime.calls), 1)

    def test_zero_elapsed_time_reports_infinite_throughput(self):
        result = TiledMatrixMultiplier(FakePhysicalRuntime()).run(
            np.ones((1, 1), dtype=np.int8),
            np.ones((1, 1), dtype=np.int8),
        )
        self.assertEqual(result.metrics.elapsed_seconds, 0.0)
        self.assertEqual(result.metrics.operations_per_second, float("inf"))


class MatrixMultiplicationNotebookTests(unittest.TestCase):
    def test_notebook_is_output_free_and_uses_only_public_runtime(self):
        notebook_path = EXAMPLE_ROOT / "matrix_multiplication.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 4)
        for cell in code_cells:
            self.assertEqual(cell.get("outputs"), [])
            self.assertIsNone(cell.get("execution_count"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for required in (
            "load_pynq_runtime",
            "TiledMatrixMultiplier",
            "np.testing.assert_array_equal",
            "non_aligned",
            "repeated",
            "operations_per_second",
        ):
            self.assertIn(required, source)
        for forbidden in (".mmio", "sendchannel", "recvchannel", "allocate("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
