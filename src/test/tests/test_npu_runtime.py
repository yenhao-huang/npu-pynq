import importlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np


class FakeBuffer:
    def __init__(self, shape, dtype, physical_address):
        self.array = np.zeros(shape, dtype=dtype)
        self.physical_address = physical_address
        self.flush_count = 0
        self.invalidate_count = 0

    @property
    def nbytes(self):
        return self.array.nbytes

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.array, dtype=dtype)

    def __setitem__(self, key, value):
        self.array[key] = value

    def flush(self):
        self.flush_count += 1

    def invalidate(self):
        self.invalidate_count += 1


class FakeAllocator:
    def __init__(self, addresses=(0x1000, 0x2000, 0x3000)):
        self.addresses = list(addresses)
        self.buffers = []

    def __call__(self, shape, dtype):
        buffer = FakeBuffer(shape, dtype, self.addresses[len(self.buffers)])
        self.buffers.append(buffer)
        return buffer


class FakeDMAMMIO:
    def __init__(self, events, reset_stuck=False):
        self.events = events
        self.registers = {}
        self.reset_stuck = reset_stuck
        self.channels = {}

    def read(self, offset):
        self.events.append(("dma_read", offset))
        return self.registers.get(offset, 0)

    def write(self, offset, value):
        self.events.append(("dma_write", offset, value))
        if value == 0x4 and not self.reset_stuck:
            self.registers[offset] = 0
            self.channels[offset].idle = True
        else:
            self.registers[offset] = value


class FakeChannel:
    def __init__(self, events, name, idle=True, mmio=None, offset=0):
        self.events = events
        self.name = name
        self.idle = idle
        self._mmio = mmio
        self._offset = offset
        self._first_transfer = True
        self.transferred = None
        self.stop_count = 0
        self.start_count = 0

    def transfer(self, buffer, nbytes=None):
        byte_count = buffer.nbytes if nbytes is None else nbytes
        self.transferred = byte_count
        self._first_transfer = False
        self.events.append((self.name, byte_count))

    def stop(self):
        self.stop_count += 1
        self.idle = True

    def start(self):
        self.start_count += 1
        self.idle = True


class FakeDMA:
    def __init__(self, events, send_idle=True, recv_idle=True, reset_stuck=False):
        self.mmio = FakeDMAMMIO(events, reset_stuck=reset_stuck)
        self.sendchannel = FakeChannel(events, "send", send_idle, self.mmio, 0x00)
        self.recvchannel = FakeChannel(events, "recv", recv_idle, self.mmio, 0x30)
        self.mmio.channels = {0x00: self.sendchannel, 0x30: self.recvchannel}


class FakeMMIO:
    def __init__(self, events, magic=0x3155504E, version=0x00010000, caps=0x1B):
        self.events = events
        self.registers = {0x00: magic, 0x04: version, 0x08: caps, 0x10: 2, 0x14: 0}

    def read(self, offset):
        self.events.append(("read", offset))
        return self.registers.get(offset, 0)

    def write(self, offset, value):
        self.events.append(("write", offset, value))
        self.registers[offset] = value


class FakeOverlay:
    def __init__(self, *, parameters=None, magic=0x3155504E, version=0x00010000,
                 caps=0x1B, send_idle=True, recv_idle=True, reset_stuck=False):
        self.events = []
        params = parameters or {"ROWS": "2", "COLUMNS": "2", "MAX_K": "256"}
        self.ip_dict = {
            "npu_matrix_accelerator_0": {
                "phys_addr": 0x43C00000, "addr_range": 0x10000, "parameters": params
            },
            "axi_dma_0": {"phys_addr": 0x40400000, "addr_range": 0x10000},
        }
        self.npu_matrix_accelerator_0 = FakeMMIO(self.events, magic, version, caps)
        self.axi_dma_0 = FakeDMA(self.events, send_idle, recv_idle, reset_stuck)


class StepClock:
    def __init__(self, step=0.001):
        self.value = 0.0
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


class NPURuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime_module = importlib.import_module("src.runtime.npu")

    def make_runtime(self, overlay=None, allocator=None, clock=None):
        return self.runtime_module.NPURuntime(
            overlay or FakeOverlay(),
            allocator=allocator or FakeAllocator(),
            monotonic=clock or StepClock(),
        )

    def test_import_does_not_require_pynq(self):
        self.assertNotIn("pynq", sys.modules)

    def test_metadata_and_abi_are_discovered(self):
        runtime = self.make_runtime()
        self.assertEqual((runtime.max_m, runtime.max_n, runtime.max_k), (2, 2, 256))
        self.assertEqual(runtime.abi_major, self.runtime_module.VERSION_MAJOR)
        self.assertEqual(runtime.capabilities, self.runtime_module.REQUIRED_CAPABILITIES)

    def test_phase_zero_abi_constants_remain_in_parity(self):
        from src.test.model import abi

        module = self.runtime_module
        self.assertEqual(module.MAGIC, abi.ABI_MAGIC)
        self.assertEqual(module.VERSION_MAJOR, abi.ABI_MAJOR)
        self.assertEqual(module.REQUIRED_CAPABILITIES, int(abi.MATRIX_REQUIRED_CAPABILITIES))
        self.assertEqual(
            [module.REG_MAGIC, module.REG_VERSION, module.REG_CAPABILITIES,
             module.REG_CONTROL, module.REG_STATUS, module.REG_ERROR,
             module.REG_M, module.REG_N, module.REG_K, module.REG_A_STRIDE,
             module.REG_B_STRIDE, module.REG_C_STRIDE, module.REG_TIMEOUT_CYCLES,
             module.REG_CYCLES_LO, module.REG_CYCLES_HI],
            [int(abi.Register[name]) for name in (
                "MAGIC", "VERSION", "CAPABILITIES", "CONTROL", "STATUS", "ERROR",
                "M", "N", "K", "A_STRIDE", "B_STRIDE", "C_STRIDE",
                "TIMEOUT_CYCLES", "CYCLES_LO", "CYCLES_HI"
            )],
        )

    def test_missing_metadata_fails_before_mmio(self):
        overlay = FakeOverlay()
        del overlay.ip_dict["axi_dma_0"]
        with self.assertRaises(self.runtime_module.MetadataError):
            self.make_runtime(overlay=overlay)
        self.assertEqual(overlay.events, [])

    def test_bad_abi_is_rejected(self):
        for kwargs in ({"magic": 0}, {"version": 0x00020000}, {"caps": 0x01}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(self.runtime_module.ABIError):
                    self.make_runtime(overlay=FakeOverlay(**kwargs))

    def test_valid_preflight_allocates_signed_dense_buffers(self):
        allocator = FakeAllocator()
        runtime = self.make_runtime(allocator=allocator)
        job = runtime.preflight(
            np.array([[-128, 127], [7, -3]], dtype=np.int8),
            np.array([[-1, 2], [4, -5]], dtype=np.int8),
        )
        self.assertEqual((job.m, job.n, job.k), (2, 2, 2))
        self.assertEqual([buffer.nbytes for buffer in allocator.buffers], [4, 4, 16])
        self.assertEqual(allocator.buffers[2].array.dtype, np.int32)

    def test_invalid_input_never_writes_or_starts_dma(self):
        overlay = FakeOverlay()
        runtime = self.make_runtime(overlay=overlay)
        initial_events = list(overlay.events)
        with self.assertRaises(self.runtime_module.ValidationError):
            runtime.run(np.ones((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.int8))
        self.assertEqual(overlay.events, initial_events)

    def test_shape_and_limit_validation(self):
        runtime = self.make_runtime()
        invalid_pairs = [
            (np.ones(2, dtype=np.int8), np.ones((2, 1), dtype=np.int8)),
            (np.ones((2, 3), dtype=np.int8), np.ones((2, 1), dtype=np.int8)),
            (np.ones((3, 1), dtype=np.int8), np.ones((1, 1), dtype=np.int8)),
        ]
        for a_matrix, b_matrix in invalid_pairs:
            with self.subTest(a=a_matrix.shape, b=b_matrix.shape):
                with self.assertRaises(self.runtime_module.ValidationError):
                    runtime.preflight(a_matrix, b_matrix)

    def test_alignment_wrap_and_alias_are_rejected(self):
        cases = [
            (0x1001, 0x2000, 0x3000),
            (0x1000, 0x2000, 0x1000),
            (0x10000000000000000, 0x2000, 0x3000),
        ]
        for addresses in cases:
            with self.subTest(addresses=addresses):
                runtime = self.make_runtime(allocator=FakeAllocator(addresses))
                with self.assertRaises(self.runtime_module.BufferError):
                    runtime.preflight(
                        np.ones((2, 2), dtype=np.int8), np.ones((2, 2), dtype=np.int8)
                    )

    def test_exact_sequence_and_signed_result(self):
        overlay = FakeOverlay()
        allocator = FakeAllocator()
        runtime = self.make_runtime(overlay=overlay, allocator=allocator)
        allocator_result = np.array([[636, -891], [-19, 29]], dtype=np.int32)

        original_transfer = overlay.axi_dma_0.recvchannel.transfer
        def receive_and_fill(buffer, nbytes=None):
            original_transfer(buffer, nbytes)
            buffer.array[:] = allocator_result
        overlay.axi_dma_0.recvchannel.transfer = receive_and_fill

        result = runtime.run(
            np.array([[-128, 127], [7, -3]], dtype=np.int8),
            np.array([[-1, 2], [4, -5]], dtype=np.int8),
            hardware_timeout_cycles=1000,
            software_timeout=1.0,
        )
        np.testing.assert_array_equal(result, allocator_result)
        important = [event for event in overlay.events if event[0] in ("recv", "send") or
                     (event[0] == "write" and event[1] == 0x0C)]
        self.assertEqual(important, [
            ("write", 0x0C, 2), ("recv", 16), ("write", 0x0C, 1),
            ("send", 4), ("send", 4),
        ])
        self.assertEqual((allocator.buffers[0].flush_count, allocator.buffers[1].flush_count), (1, 1))
        self.assertEqual(allocator.buffers[2].invalidate_count, 1)

    def test_success_exposes_stable_cycle_metrics_across_rollover(self):
        overlay = FakeOverlay()
        mmio = overlay.npu_matrix_accelerator_0
        high_values = iter((1, 2, 2, 2))
        low_values = iter((0xFFFFFFFE, 3))
        original_read = mmio.read

        def rollover_read(offset):
            if offset == self.runtime_module.REG_CYCLES_HI:
                mmio.events.append(("read", offset))
                return next(high_values)
            if offset == self.runtime_module.REG_CYCLES_LO:
                mmio.events.append(("read", offset))
                return next(low_values)
            return original_read(offset)

        mmio.read = rollover_read
        runtime = self.make_runtime(overlay=overlay)
        result = runtime.run(
            np.ones((1, 1), dtype=np.int8),
            np.ones((1, 1), dtype=np.int8),
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(runtime.last_metrics.cycles, (2 << 32) | 3)

    def test_failed_job_clears_prior_cycle_metrics(self):
        overlay = FakeOverlay()
        runtime = self.make_runtime(overlay=overlay)
        runtime.last_metrics = SimpleNamespace(cycles=123)
        overlay.npu_matrix_accelerator_0.registers[0x10] = 4
        overlay.npu_matrix_accelerator_0.registers[0x14] = 4
        with self.assertRaises(self.runtime_module.HardwareError):
            runtime.run(
                np.ones((1, 1), dtype=np.int8),
                np.ones((1, 1), dtype=np.int8),
            )
        self.assertIsNone(runtime.last_metrics)

    def test_cycle_rollover_timeout_recovers_without_metrics(self):
        overlay = FakeOverlay()
        mmio = overlay.npu_matrix_accelerator_0
        original_read = mmio.read
        high = 0

        def never_stable(offset):
            nonlocal high
            if offset == self.runtime_module.REG_CYCLES_HI:
                high += 1
                mmio.events.append(("read", offset))
                return high
            if offset == self.runtime_module.REG_CYCLES_LO:
                mmio.events.append(("read", offset))
                return 0
            return original_read(offset)

        mmio.read = never_stable
        runtime = self.make_runtime(
            overlay=overlay, clock=StepClock(step=0.25)
        )
        with self.assertRaisesRegex(TimeoutError, "cycle counter"):
            runtime.run(
                np.ones((1, 1), dtype=np.int8),
                np.ones((1, 1), dtype=np.int8),
                software_timeout=0.5,
            )
        self.assertIsNone(runtime.last_metrics)
        self.assertGreaterEqual(
            overlay.events.count(("write", self.runtime_module.REG_CONTROL, 2)),
            2,
        )

    def test_hardware_error_is_typed_and_recovers(self):
        overlay = FakeOverlay()
        overlay.npu_matrix_accelerator_0.registers[0x10] = 4
        overlay.npu_matrix_accelerator_0.registers[0x14] = 4
        runtime = self.make_runtime(overlay=overlay)
        with self.assertRaises(self.runtime_module.HardwareError) as raised:
            runtime.run(np.ones((1, 1), dtype=np.int8), np.ones((1, 1), dtype=np.int8))
        self.assertEqual((raised.exception.code, raised.exception.name), (4, "STREAM_LENGTH"))
        self.assertIn(("write", 0x0C, 2), overlay.events)

    def test_dma_timeout_is_finite_and_recovers(self):
        overlay = FakeOverlay(send_idle=False)
        runtime = self.make_runtime(
            overlay=overlay,
            allocator=FakeAllocator(
                (0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000)
            ),
            clock=StepClock(step=0.1),
        )
        with self.assertRaises(TimeoutError):
            runtime.run(
                np.ones((1, 1), dtype=np.int8), np.ones((1, 1), dtype=np.int8),
                software_timeout=0.2,
            )
        for channel in (
            overlay.axi_dma_0.sendchannel,
            overlay.axi_dma_0.recvchannel,
        ):
            self.assertEqual(channel.stop_count, 0)
            self.assertEqual(channel.start_count, 0)
            self.assertTrue(channel._first_transfer)
        self.assertIn(("dma_write", 0x00, 0x4), overlay.events)
        self.assertNotIn(("dma_write", 0x30, 0x4), overlay.events)
        self.assertIn(("dma_write", 0x00, 0x1), overlay.events)
        self.assertIn(("dma_write", 0x30, 0x1), overlay.events)
        recovered = runtime.run(
            np.ones((1, 1), dtype=np.int8),
            np.ones((1, 1), dtype=np.int8),
        )
        self.assertEqual((recovered.shape, recovered.dtype), ((1, 1), np.int32))

    def test_stuck_dma_reset_recovery_is_bounded(self):
        overlay = FakeOverlay(send_idle=False, reset_stuck=True)
        runtime = self.make_runtime(overlay=overlay, clock=StepClock(step=0.1))
        with self.assertRaises(TimeoutError):
            runtime.run(
                np.ones((1, 1), dtype=np.int8),
                np.ones((1, 1), dtype=np.int8),
                software_timeout=0.2,
            )
        self.assertLess(
            overlay.events.count(("dma_read", 0x00)),
            20,
        )
        self.assertNotIn(("dma_write", 0x00, 0x1), overlay.events)
        for channel in (
            overlay.axi_dma_0.sendchannel,
            overlay.axi_dma_0.recvchannel,
        ):
            self.assertEqual(channel.stop_count, 0)
            self.assertEqual(channel.start_count, 0)

    def test_dma_length_mismatch_never_returns_result(self):
        overlay = FakeOverlay()
        runtime = self.make_runtime(overlay=overlay)
        channel = overlay.axi_dma_0.recvchannel
        original_transfer = channel.transfer
        def wrong_length(buffer, nbytes=None):
            original_transfer(buffer, nbytes)
            channel.transferred = 0
        channel.transfer = wrong_length
        with self.assertRaises(self.runtime_module.DMAError):
            runtime.run(np.ones((1, 1), dtype=np.int8), np.ones((1, 1), dtype=np.int8))


if __name__ == "__main__":
    unittest.main()
