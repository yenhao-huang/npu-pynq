import unittest

from src.test import model
from src.vendors.npu_pynq.spec import abi


class PublicContractApiTests(unittest.TestCase):
    def test_required_phase1_symbols_are_exported(self):
        # src.test.model is the numeric golden reference.
        required = {
            "ArrayConfiguration",
            "PerformanceAssumptions",
            "estimate_matmul",
            "mac_int8_int32",
            "matmul_int8",
            "requantize_int32_to_int8",
        }
        self.assertTrue(required.issubset(set(model.__all__)))
        for symbol in required:
            with self.subTest(symbol=symbol):
                self.assertTrue(hasattr(model, symbol))

    def test_the_abi_lives_in_the_vendor_spec(self):
        for symbol in ("ABI_MAGIC", "AbiVersion", "Capability", "IsaRegister", "Register",
                       "REQUIRED_CAPABILITIES", "negotiate_abi"):
            with self.subTest(symbol=symbol):
                self.assertTrue(hasattr(abi, symbol))
        self.assertFalse(hasattr(model, "Register"))


if __name__ == "__main__":
    unittest.main()
