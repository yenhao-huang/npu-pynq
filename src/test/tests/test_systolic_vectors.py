import tempfile
import unittest
from pathlib import Path

from src.test.model import matmul_int8
from src.test.vectors.generate_systolic_vectors import (
    DEFAULT_CASES,
    DEFAULT_SEED,
    MAX_K,
    PHYSICAL_COLUMNS,
    PHYSICAL_ROWS,
    build_vectors,
    write_vectors,
)


FIXTURE = Path("src/test/vectors/systolic_2x2_k8.txt")


class SystolicVectorTests(unittest.TestCase):
    def test_tracked_fixture_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated = Path(temporary_directory) / "vectors.txt"
            write_vectors(generated)
            self.assertEqual(generated.read_bytes(), FIXTURE.read_bytes())

    def test_case_count_seed_and_endpoint_coverage(self):
        vectors = build_vectors()
        self.assertEqual(len(vectors), DEFAULT_CASES)
        self.assertEqual(DEFAULT_SEED, 0x5A17)
        values = {
            value
            for vector in vectors
            for value in vector.a_padded + vector.b_padded
        }
        self.assertTrue({-128, -1, 0, 1, 127}.issubset(values))

    def test_every_expected_result_matches_golden_model(self):
        for case_index, vector in enumerate(build_vectors()):
            matrix_a = tuple(
                tuple(
                    vector.a_padded[row * MAX_K + reduction]
                    for reduction in range(vector.k)
                )
                for row in range(vector.m)
            )
            matrix_b = tuple(
                tuple(
                    vector.b_padded[
                        reduction * PHYSICAL_COLUMNS + column
                    ]
                    for column in range(vector.n)
                )
                for reduction in range(vector.k)
            )
            expected = matmul_int8(matrix_a, matrix_b)
            padded = tuple(
                expected[row][column]
                if row < vector.m and column < vector.n
                else 0
                for row in range(PHYSICAL_ROWS)
                for column in range(PHYSICAL_COLUMNS)
            )
            with self.subTest(case_index=case_index):
                self.assertEqual(vector.c_padded, padded)


if __name__ == "__main__":
    unittest.main()
