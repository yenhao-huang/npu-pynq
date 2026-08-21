import unittest

from src.test.model.numeric import (
    INT32_MAX,
    INT32_MIN,
    mac_int8_int32,
    matmul_int8,
    requantize_int32_to_int8,
    saturate_int8,
    saturate_int32,
)


class SaturationTests(unittest.TestCase):
    def test_signed_endpoints_multiply_as_signed_values(self):
        endpoints = (-128, -1, 0, 1, 127)
        for lhs in endpoints:
            for rhs in endpoints:
                with self.subTest(lhs=lhs, rhs=rhs):
                    self.assertEqual(mac_int8_int32(0, lhs, rhs), lhs * rhs)

    def test_accumulator_saturates_after_each_mac(self):
        self.assertEqual(mac_int8_int32(INT32_MAX, 1, 1), INT32_MAX)
        self.assertEqual(mac_int8_int32(INT32_MIN, -1, 1), INT32_MIN)
        self.assertEqual(saturate_int32(INT32_MAX + 100), INT32_MAX)
        self.assertEqual(saturate_int32(INT32_MIN - 100), INT32_MIN)

    def test_int8_saturation(self):
        self.assertEqual(saturate_int8(-129), -128)
        self.assertEqual(saturate_int8(-128), -128)
        self.assertEqual(saturate_int8(127), 127)
        self.assertEqual(saturate_int8(128), 127)

    def test_invalid_operands_are_rejected(self):
        for invalid in (-129, 128):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    mac_int8_int32(0, invalid, 1)


class RequantizationTests(unittest.TestCase):
    def test_halfway_values_round_away_from_zero(self):
        half_q31 = 1 << 30
        self.assertEqual(requantize_int32_to_int8(1, half_q31), 1)
        self.assertEqual(requantize_int32_to_int8(-1, half_q31), -1)
        self.assertEqual(requantize_int32_to_int8(3, half_q31), 2)
        self.assertEqual(requantize_int32_to_int8(-3, half_q31), -2)

    def test_zero_point_and_output_saturation(self):
        almost_one_q31 = INT32_MAX
        self.assertEqual(
            requantize_int32_to_int8(INT32_MAX, almost_one_q31), 127
        )
        self.assertEqual(
            requantize_int32_to_int8(INT32_MIN, almost_one_q31), -128
        )
        self.assertEqual(requantize_int32_to_int8(0, 0, zero_point=7), 7)

    def test_parameter_ranges_are_enforced(self):
        with self.assertRaises(ValueError):
            requantize_int32_to_int8(0, 0, shift=-1)
        with self.assertRaises(ValueError):
            requantize_int32_to_int8(0, 0, shift=32)
        with self.assertRaises(ValueError):
            requantize_int32_to_int8(0, 1 << 31)
        with self.assertRaises(ValueError):
            requantize_int32_to_int8(0, 0, zero_point=128)


class MatrixTests(unittest.TestCase):
    @staticmethod
    def independent_reference(lhs, rhs):
        rows = len(lhs)
        reduction = len(lhs[0])
        columns = len(rhs[0])
        output = []
        for row in range(rows):
            output_row = []
            for column in range(columns):
                accumulator = 0
                for index in range(reduction):
                    accumulator += lhs[row][index] * rhs[index][column]
                    accumulator = max(INT32_MIN, min(INT32_MAX, accumulator))
                output_row.append(accumulator)
            output.append(tuple(output_row))
        return tuple(output)

    def test_non_square_row_major_fixture(self):
        lhs = ((1, -2, 3), (4, 5, -6))
        rhs = ((7, 8), (-9, 10), (11, -12))
        self.assertEqual(matmul_int8(lhs, rhs), self.independent_reference(lhs, rhs))

    def test_signed_endpoint_fixture(self):
        lhs = ((-128, 127), (1, -1))
        rhs = ((127, -128, 1), (-128, 127, -1))
        self.assertEqual(matmul_int8(lhs, rhs), self.independent_reference(lhs, rhs))

    def test_each_job_starts_from_zero(self):
        lhs = ((127,),)
        rhs = ((127,),)
        self.assertEqual(matmul_int8(lhs, rhs), ((16129,),))
        self.assertEqual(matmul_int8(((0,),), rhs), ((0,),))

    def test_invalid_shapes_and_values_are_rejected(self):
        invalid_cases = (
            ((), ((1,),)),
            (((1, 2),), ((1,),)),
            (((1,), (2, 3)), ((1,),)),
            (((128,),), ((1,),)),
        )
        for lhs, rhs in invalid_cases:
            with self.subTest(lhs=lhs, rhs=rhs):
                with self.assertRaises(ValueError):
                    matmul_int8(lhs, rhs)


if __name__ == "__main__":
    unittest.main()
