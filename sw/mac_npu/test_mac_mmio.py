import sys
import unittest
from pathlib import Path


SERVICE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVICE_DIR))

from mac_mmio import (  # noqa: E402
    ACCUMULATOR,
    CONTROL,
    CONTROL_CLEAR,
    CONTROL_START,
    OPERAND_A,
    OPERAND_B,
    STATUS,
    MacMMIO,
)


class FakeMacMMIO:
    def __init__(self):
        self.registers = {ACCUMULATOR: 0, STATUS: 0}

    def write(self, offset, value):
        if offset in (OPERAND_A, OPERAND_B):
            self.registers[offset] = value & 0xFF
        elif offset == CONTROL and value & CONTROL_CLEAR:
            self.registers[ACCUMULATOR] = 0
            self.registers[STATUS] = 0
        elif offset == CONTROL and value & CONTROL_START:
            a = self._signed(self.registers[OPERAND_A], 8)
            b = self._signed(self.registers[OPERAND_B], 8)
            result = (self.registers[ACCUMULATOR] + a * b) & 0xFFFFFFFF
            self.registers[ACCUMULATOR] = result
            self.registers[STATUS] = 1

    def read(self, offset):
        return self.registers.get(offset, 0)

    @staticmethod
    def _signed(value, width):
        return value - (1 << width) if value & (1 << (width - 1)) else value


class MacMMIOTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeMacMMIO()
        self.mac = MacMMIO(self.fake)

    def test_clear_start_accumulate_and_signed_readback(self):
        self.mac.clear()
        self.assertEqual(self.mac.mac(2, 3, timeout=0.01), 6)
        self.assertEqual(self.mac.mac(-7, 6, timeout=0.01), -36)
        self.mac.clear()
        self.assertEqual(self.mac.read_accumulator(), 0)

    def test_signed_boundary_operands(self):
        self.assertEqual(self.mac.mac(-128, -128, timeout=0.01), 16384)
        self.assertEqual(self.mac.mac(127, -128, timeout=0.01), 128)

    def test_rejects_out_of_range_operand(self):
        with self.assertRaises(ValueError):
            self.mac.mac(128, 0)

    def test_timeout_when_done_never_asserts(self):
        self.fake.write = lambda offset, value: None
        with self.assertRaises(TimeoutError):
            self.mac.mac(1, 1, timeout=0.001)


if __name__ == "__main__":
    unittest.main()
