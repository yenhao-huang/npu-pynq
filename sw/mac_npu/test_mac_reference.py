import random
import sys
import unittest
from pathlib import Path


SERVICE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVICE_DIR))

from mac_reference import dot_product, mac_step, wrap_signed  # noqa: E402


class MacReferenceTests(unittest.TestCase):
    def test_directed_vector(self):
        self.assertEqual(dot_product([2, 4], [3, 5]), 26)

    def test_signed_boundaries(self):
        self.assertEqual(mac_step(0, -128, -128), 16384)
        self.assertEqual(mac_step(0, 127, -128), -16256)

    def test_accumulator_wrap(self):
        self.assertEqual(mac_step((1 << 31) - 1, 1, 1), -(1 << 31))
        self.assertEqual(wrap_signed(-(1 << 31) - 1, 32), (1 << 31) - 1)

    def test_random_vectors_match_python_sum(self):
        rng = random.Random(0x4D4143)
        a_values = [rng.randint(-128, 127) for _ in range(256)]
        b_values = [rng.randint(-128, 127) for _ in range(256)]
        expected = wrap_signed(sum(a * b for a, b in zip(a_values, b_values)), 32)
        self.assertEqual(dot_product(a_values, b_values), expected)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            mac_step(0, 128, 1)
        with self.assertRaises(ValueError):
            dot_product([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
