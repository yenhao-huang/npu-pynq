import unittest

from pathlib import Path
import re

from src.vendors.npu_pynq.spec.abi import (
    ABI_MAGIC,
    ABI_WINDOW_BYTES,
    REQUIRED_CAPABILITIES,
    AbiCompatibilityError,
    AbiVersion,
    Capability,
    Control,
    ErrorCode,
    IsaRegister,
    IsaStatus,
    Register,
    Status,
    negotiate_abi,
)

ROOT = Path(__file__).resolve().parents[3]


class ConstantTests(unittest.TestCase):
    def test_identity_and_register_map_are_exact(self):
        self.assertEqual(ABI_MAGIC, 0x3155504E)
        self.assertEqual(ABI_WINDOW_BYTES, 0x100)
        expected = {
            "MAGIC": 0x00,
            "VERSION": 0x04,
            "CAPABILITIES": 0x08,
            "CONTROL": 0x0C,
            "STATUS": 0x10,
            "ERROR": 0x14,
            "M": 0x18,
            "N": 0x1C,
            "K": 0x20,
            "A_STRIDE": 0x24,
            "B_STRIDE": 0x28,
            "C_STRIDE": 0x2C,
            "TIMEOUT_CYCLES": 0x30,
            "CYCLES_LO": 0x34,
            "CYCLES_HI": 0x38,
        }
        self.assertEqual({item.name: item.value for item in Register}, expected)
        self.assertEqual(len({item.value for item in Register}), len(Register))

    def test_control_status_and_errors_are_exact(self):
        self.assertEqual(Control.START, 1)
        self.assertEqual(Control.SOFT_RESET, 2)
        self.assertEqual(Status.BUSY, 1)
        self.assertEqual(Status.DONE, 2)
        self.assertEqual(Status.ERROR, 4)
        self.assertEqual(
            {item.name: item.value for item in ErrorCode},
            {
                "NONE": 0,
                "INVALID_DIMENSION": 1,
                "INVALID_STRIDE": 2,
                "BUSY_START": 3,
                "STREAM_LENGTH": 4,
                "TIMEOUT": 5,
                "INVALID_TIMEOUT": 6,
                "INTERNAL": 255,
            },
        )


class CompatibilityTests(unittest.TestCase):
    def test_version_round_trip(self):
        version = AbiVersion(major=1, minor=7)
        self.assertEqual(AbiVersion.decode(version.encode()), version)
        self.assertEqual(version.encode(), 0x00010007)

    def test_newer_minor_with_required_capabilities_is_compatible(self):
        version = negotiate_abi(
            magic=ABI_MAGIC,
            version_word=AbiVersion(1, 9).encode(),
            capabilities=int(REQUIRED_CAPABILITIES | Capability.REQUANT_INT8),
        )
        self.assertEqual(version, AbiVersion(1, 9))

    def test_bad_magic_major_or_capabilities_are_rejected(self):
        cases = (
            (0, AbiVersion(1, 0).encode(), int(REQUIRED_CAPABILITIES)),
            (ABI_MAGIC, AbiVersion(2, 0).encode(), int(REQUIRED_CAPABILITIES)),
            (ABI_MAGIC, AbiVersion(1, 0).encode(), int(Capability.MATRIX_INT8)),
            (ABI_MAGIC, AbiVersion(1, 1).encode(), int(REQUIRED_CAPABILITIES & ~Capability.ISA_FRONTEND)),
        )
        for magic, version, capabilities in cases:
            with self.subTest(magic=magic, version=version, capabilities=capabilities):
                with self.assertRaises(AbiCompatibilityError):
                    negotiate_abi(magic, version, capabilities)


class PipelinedJobAbiTest(unittest.TestCase):
    def test_pipelined_jobs_is_advertised_not_required(self):
        self.assertNotIn(Capability.PIPELINED_JOBS, REQUIRED_CAPABILITIES)
        version = negotiate_abi(
            magic=ABI_MAGIC,
            version_word=AbiVersion(1, 0).encode(),
            capabilities=int(REQUIRED_CAPABILITIES),
        )
        self.assertEqual(version, AbiVersion(1, 0))

    def test_pipelined_device_still_negotiates(self):
        version = negotiate_abi(
            magic=ABI_MAGIC,
            version_word=AbiVersion(1, 0).encode(),
            capabilities=int(REQUIRED_CAPABILITIES | Capability.PIPELINED_JOBS),
        )
        self.assertEqual(version, AbiVersion(1, 0))

    def test_accept_occupies_the_reserved_status_bit(self):
        self.assertEqual(int(Status.ACCEPT), 1 << 3)
        self.assertEqual(
            int(Status.BUSY | Status.DONE | Status.ERROR | Status.ACCEPT), 0xF
        )


class DriverMatchesSpecTests(unittest.TestCase):
    """The board driver hard-codes the register map in C++; it must equal the spec."""

    def test_the_hardware_reports_what_the_runtime_requires(self):
        regs = (ROOT / "src/hw/rtl/npu_matrix/npu_accelerator/csr/npu_csr.sv").read_text()
        reported = int(re.search(r"ABI_CAPABILITIES = 32'h([0-9a-fA-F]+)", regs).group(1), 16)
        self.assertEqual(int(REQUIRED_CAPABILITIES) & ~reported, 0)
        version = negotiate_abi(ABI_MAGIC, AbiVersion().encode(), reported)
        self.assertEqual(version, AbiVersion())

    def test_npu_rt_registers_and_status_bits(self):
        source = (ROOT / "src/vendors/npu_pynq/driver/npu_rt/pynq.cpp").read_text()
        found = {name: int(value, 16) for name, value in re.findall(r"REG_(\w+) = (0x[0-9A-Fa-f]+)", source)}
        self.assertTrue(found, "no REG_* constants found in pynq.cpp")
        isa = {member.name: member.value for member in IsaRegister}
        for name, value in found.items():
            key = name[4:] if name.startswith("ISA_") else name
            with self.subTest(register=name):
                self.assertEqual(value, isa[key])
        bits = {name: int(value) for name, value in re.findall(r"STATUS_(\w+) = (\d+)", source)}
        self.assertEqual(bits, {member.name: member.value for member in IsaStatus})

    def test_the_runtime_takes_the_register_map_from_the_spec(self):
        source = (ROOT / "src/runtime/delegate/npu.py").read_text()
        self.assertIn("from src.vendors.npu_pynq.spec.abi import", source)
        self.assertNotRegex(source, r"REG_\w+ = 0x")


if __name__ == "__main__":
    unittest.main()
