import unittest

from src.test import model


class PublicContractApiTests(unittest.TestCase):
    def test_required_phase1_symbols_are_exported(self):
        required = {
            "ABI_MAGIC",
            "AbiVersion",
            "ArrayConfiguration",
            "BufferRange",
            "MatrixBuffers",
            "MatrixJob",
            "PerformanceAssumptions",
            "Register",
            "estimate_matmul",
            "mac_int8_int32",
            "matmul_int8",
            "requantize_int32_to_int8",
        }
        self.assertTrue(required.issubset(set(model.__all__)))
        for symbol in required:
            with self.subTest(symbol=symbol):
                self.assertTrue(hasattr(model, symbol))


if __name__ == "__main__":
    unittest.main()
