import unittest

from src.model import numeric as production_numeric
from src.test.model import numeric as compatibility_numeric


class ProductionNumericContractTests(unittest.TestCase):
    def test_test_package_reexports_production_contract(self):
        exported = (
            "INT8_MIN",
            "INT8_MAX",
            "INT32_MIN",
            "INT32_MAX",
            "saturate_int8",
            "saturate_int32",
            "saturating_add_int32",
            "mac_int8_int32",
            "round_ratio_away_from_zero",
            "requantize_int32_to_int8",
            "matmul_int8",
        )
        for name in exported:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(compatibility_numeric, name),
                    getattr(production_numeric, name),
                )

    def test_signed_endpoints_rounding_and_saturation(self):
        self.assertEqual(production_numeric.mac_int8_int32(0, -128, 127), -16256)
        self.assertEqual(
            production_numeric.saturating_add_int32(
                production_numeric.INT32_MAX, 1
            ),
            production_numeric.INT32_MAX,
        )
        self.assertEqual(
            production_numeric.saturating_add_int32(
                production_numeric.INT32_MIN, -1
            ),
            production_numeric.INT32_MIN,
        )
        self.assertEqual(production_numeric.round_ratio_away_from_zero(1, 2), 1)
        self.assertEqual(production_numeric.round_ratio_away_from_zero(-1, 2), -1)
        self.assertEqual(production_numeric.round_ratio_away_from_zero(3, 2), 2)
        self.assertEqual(production_numeric.round_ratio_away_from_zero(-3, 2), -2)

    def test_invalid_addend_and_denominator_fail(self):
        with self.assertRaises(TypeError):
            production_numeric.saturating_add_int32(0, True)
        with self.assertRaises(ValueError):
            production_numeric.round_ratio_away_from_zero(1, 0)


if __name__ == "__main__":
    unittest.main()
