import unittest

from src.test.model.abi import (
    ABI_MAGIC,
    ABI_WINDOW_BYTES,
    MATRIX_REQUIRED_CAPABILITIES,
    AbiCompatibilityError,
    AbiVersion,
    BufferRange,
    Capability,
    Control,
    ErrorCode,
    JobValidationError,
    MatrixBuffers,
    MatrixJob,
    Register,
    Status,
    negotiate_abi,
)


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
            "JOB_FLAGS": 0x3C,
            "OUTPUT_ZERO_POINT": 0x40,
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
                "INVALID_REQUANTIZATION": 7,
                "INTERNAL": 255,
            },
        )


class CompatibilityTests(unittest.TestCase):
    def test_version_round_trip(self):
        version = AbiVersion(major=2, minor=7)
        self.assertEqual(AbiVersion.decode(version.encode()), version)
        self.assertEqual(version.encode(), 0x00020007)

    def test_newer_minor_with_required_capabilities_is_compatible(self):
        version = negotiate_abi(
            magic=ABI_MAGIC,
            version_word=AbiVersion(2, 9).encode(),
            capabilities=int(MATRIX_REQUIRED_CAPABILITIES | Capability.REQUANT_INT8),
        )
        self.assertEqual(version, AbiVersion(2, 9))

    def test_bad_magic_major_or_capabilities_are_rejected(self):
        cases = (
            (0, AbiVersion(2, 0).encode(), int(MATRIX_REQUIRED_CAPABILITIES)),
            (ABI_MAGIC, AbiVersion(1, 0).encode(), int(MATRIX_REQUIRED_CAPABILITIES)),
            (ABI_MAGIC, AbiVersion(2, 0).encode(), int(Capability.MATRIX_INT8)),
        )
        for magic, version, capabilities in cases:
            with self.subTest(magic=magic, version=version, capabilities=capabilities):
                with self.assertRaises(AbiCompatibilityError):
                    negotiate_abi(magic, version, capabilities)


class JobTests(unittest.TestCase):
    def test_dense_job_and_payload_counts(self):
        job = MatrixJob.dense(m=2, n=3, k=4, timeout_cycles=1000)
        self.assertEqual((job.a_stride, job.b_stride, job.c_stride), (4, 3, 3))
        self.assertEqual(job.input_elements, 20)
        self.assertEqual(job.output_elements, 6)
        self.assertEqual(job.payload_bytes, (8, 12, 6))

    def test_job_is_immutable(self):
        job = MatrixJob.dense(1, 1, 1, timeout_cycles=1)
        with self.assertRaises((AttributeError, TypeError)):
            job.m = 2

    def test_invalid_dimensions_strides_and_timeout_are_rejected(self):
        invalid_arguments = (
            dict(m=0, n=1, k=1, a_stride=1, b_stride=1, c_stride=4, timeout_cycles=1),
            dict(m=1, n=65536, k=1, a_stride=1, b_stride=65536, c_stride=262144, timeout_cycles=1),
            dict(m=1, n=2, k=3, a_stride=2, b_stride=2, c_stride=8, timeout_cycles=1),
            dict(m=1, n=2, k=3, a_stride=3, b_stride=1, c_stride=8, timeout_cycles=1),
            dict(m=1, n=2, k=3, a_stride=3, b_stride=2, c_stride=1, timeout_cycles=1),
            dict(m=1, n=1, k=1, a_stride=1, b_stride=1, c_stride=4, timeout_cycles=0),
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    MatrixJob(**arguments)

    def test_preflight_errors_carry_abi_codes(self):
        cases = (
            (
                dict(m=0, n=1, k=1, a_stride=1, b_stride=1, c_stride=4, timeout_cycles=1),
                ErrorCode.INVALID_DIMENSION,
            ),
            (
                dict(m=1, n=2, k=3, a_stride=2, b_stride=2, c_stride=8, timeout_cycles=1),
                ErrorCode.INVALID_STRIDE,
            ),
            (
                dict(m=1, n=1, k=1, a_stride=1, b_stride=1, c_stride=4, timeout_cycles=0),
                ErrorCode.INVALID_TIMEOUT,
            ),
        )
        for arguments, expected_code in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(JobValidationError) as raised:
                    MatrixJob(**arguments)
                self.assertEqual(raised.exception.code, expected_code)


class BufferTests(unittest.TestCase):
    def setUp(self):
        self.job = MatrixJob.dense(2, 3, 4, timeout_cycles=1000)

    def test_valid_buffers_cover_dense_payloads(self):
        buffers = MatrixBuffers(
            a=BufferRange(0x1000, 8),
            b=BufferRange(0x2000, 12),
            c=BufferRange(0x3000, 24),
        )
        buffers.validate_for(self.job)

    def test_last_aligned_32_bit_range_is_valid(self):
        BufferRange(0xFFFFFFC0, 64)

    def test_misaligned_wrapping_or_undersized_buffers_are_rejected(self):
        with self.assertRaises(ValueError):
            BufferRange(0x1001, 64)
        with self.assertRaises(ValueError):
            BufferRange(0xFFFFFFC0, 65)
        with self.assertRaises(ValueError):
            MatrixBuffers(
                a=BufferRange(0x1000, 7),
                b=BufferRange(0x2000, 12),
                c=BufferRange(0x3000, 24),
            ).validate_for(self.job)

    def test_output_must_not_overlap_an_input(self):
        with self.assertRaises(ValueError):
            MatrixBuffers(
                a=BufferRange(0x1000, 128),
                b=BufferRange(0x2000, 64),
                c=BufferRange(0x1040, 64),
            ).validate_for(self.job)


if __name__ == "__main__":
    unittest.main()
