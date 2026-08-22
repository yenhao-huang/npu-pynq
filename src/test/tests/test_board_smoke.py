import unittest

import numpy as np

from src.runtime.board_smoke import PASS_MARKER, execute_smoke


class FakeMMIO:
    def __init__(self):
        self.values = {0x10: 2, 0x14: 0, 0x34: 123, 0x38: 0}

    def read(self, offset):
        return self.values[offset]


class FakeRuntime:
    def __init__(self, result):
        self.result = result
        self.mmio = FakeMMIO()
        self.calls = []

    def run(self, matrix_a, matrix_b, **kwargs):
        self.calls.append((matrix_a.copy(), matrix_b.copy(), kwargs))
        return self.result.copy()


class BoardSmokeTests(unittest.TestCase):
    def test_exact_matrix_and_evidence(self):
        runtime = FakeRuntime(np.array([[636, -891], [-19, 29]], dtype=np.int32))
        evidence = execute_smoke(runtime, {"source_commit": "abc", "target_part": "xc7z020clg400-1"})
        self.assertEqual(evidence["pass_marker"], PASS_MARKER)
        self.assertEqual(evidence["abi"], {"status": 2, "error": 0, "cycles": 123})
        matrix_a, matrix_b, options = runtime.calls[0]
        np.testing.assert_array_equal(matrix_a, [[-128, 127], [7, -3]])
        np.testing.assert_array_equal(matrix_b, [[-1, 2], [4, -5]])
        self.assertGreater(options["hardware_timeout_cycles"], 0)
        self.assertGreater(options["software_timeout"], 0)

    def test_result_mismatch_fails_without_pass_evidence(self):
        runtime = FakeRuntime(np.zeros((2, 2), dtype=np.int32))
        with self.assertRaises(RuntimeError):
            execute_smoke(runtime, {})

    def test_status_or_cycle_failure_is_rejected(self):
        runtime = FakeRuntime(np.array([[636, -891], [-19, 29]], dtype=np.int32))
        runtime.mmio.values[0x34] = 0
        with self.assertRaises(RuntimeError):
            execute_smoke(runtime, {})


if __name__ == "__main__":
    unittest.main()
