import importlib
import sys
import unittest

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


class FakeChannel:
    def __init__(self, events, name, idle=True):
        self.events = events
        self.name = name
        self.idle = idle
        self.transferred = None
        self.stop_count = 0

    def transfer(self, buffer, nbytes=None):
        byte_count = buffer.nbytes if nbytes is None else nbytes
        self.transferred = byte_count
        self.events.append((self.name, byte_count))

    def stop(self):
        self.stop_count += 1
        self.idle = True


class PynqLikeChannel(FakeChannel):
    """Model PYNQ 3.1 simple-DMA completion bookkeeping."""

    def __init__(self, events, name, idle=True):
        super().__init__(events, name, idle)
        self.wait_count = 0
        self._requested_bytes = None

    def transfer(self, buffer, nbytes=None):
        self._requested_bytes = buffer.nbytes if nbytes is None else nbytes
        self.transferred = 0
        self.events.append((self.name, self._requested_bytes))

    def wait(self):
        self.wait_count += 1
        self.transferred = self._requested_bytes


class FakeDMA:
    def __init__(self, events, send_idle=True, recv_idle=True):
        self.sendchannel = FakeChannel(events, "send", send_idle)
        self.recvchannel = FakeChannel(events, "recv", recv_idle)


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
                 caps=0x1B, send_idle=True, recv_idle=True):
        self.events = []
        params = parameters or {"ROWS": "2", "COLUMNS": "2", "MAX_K": "256"}
        self.ip_dict = {
            "npu_matrix_accelerator_0": {
                "phys_addr": 0x43C00000, "addr_range": 0x10000, "parameters": params
            },
            "axi_dma_0": {"phys_addr": 0x40400000, "addr_range": 0x10000},
        }
        self.npu_matrix_accelerator_0 = FakeMMIO(self.events, magic, version, caps)
        self.axi_dma_0 = FakeDMA(self.events, send_idle, recv_idle)


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
             module.REG_B_STRIDE, module.REG_C_STRIDE, module.REG_TIMEOUT_CYCLES],
            [int(abi.Register[name]) for name in (
                "MAGIC", "VERSION", "CAPABILITIES", "CONTROL", "STATUS", "ERROR",
                "M", "N", "K", "A_STRIDE", "B_STRIDE", "C_STRIDE", "TIMEOUT_CYCLES"
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
        runtime = self.make_runtime(overlay=overlay, clock=StepClock(step=0.1))
        with self.assertRaises(TimeoutError):
            runtime.run(
                np.ones((1, 1), dtype=np.int8), np.ones((1, 1), dtype=np.int8),
                software_timeout=0.2,
            )
        self.assertEqual(overlay.axi_dma_0.sendchannel.stop_count, 0)
        self.assertGreaterEqual(overlay.axi_dma_0.recvchannel.stop_count, 1)

    def test_idle_pynq_channel_wait_finalizes_transferred_length(self):
        overlay = FakeOverlay()
        send = PynqLikeChannel(overlay.events, "send")
        recv = PynqLikeChannel(overlay.events, "recv")
        overlay.axi_dma_0.sendchannel = send
        overlay.axi_dma_0.recvchannel = recv
        allocator = FakeAllocator()
        expected = np.array([[636, -891], [-19, 29]], dtype=np.int32)

        original_receive = recv.transfer
        def receive_and_fill(buffer, nbytes=None):
            original_receive(buffer, nbytes)
            buffer.array[:] = expected
        recv.transfer = receive_and_fill

        result = self.make_runtime(overlay=overlay, allocator=allocator).run(
            np.array([[-128, 127], [7, -3]], dtype=np.int8),
            np.array([[-1, 2], [4, -5]], dtype=np.int8),
        )

        np.testing.assert_array_equal(result, expected)
        self.assertEqual(send.wait_count, 2)
        self.assertEqual(recv.wait_count, 1)

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
