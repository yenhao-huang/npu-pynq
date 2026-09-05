"""Bounded Phase 1B matrix runtime for a DMA-connected PYNQ overlay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np


MAGIC = 0x3155504E
VERSION_MAJOR = 2
REQUIRED_CAPABILITIES = 0x1F

REG_MAGIC = 0x00
REG_VERSION = 0x04
REG_CAPABILITIES = 0x08
REG_CONTROL = 0x0C
REG_STATUS = 0x10
REG_ERROR = 0x14
REG_M = 0x18
REG_N = 0x1C
REG_K = 0x20
REG_A_STRIDE = 0x24
REG_B_STRIDE = 0x28
REG_C_STRIDE = 0x2C
REG_TIMEOUT_CYCLES = 0x30
REG_CYCLES_LO = 0x34
REG_CYCLES_HI = 0x38
REG_JOB_FLAGS = 0x3C
REG_OUTPUT_ZERO_POINT = 0x40

JOB_FIRST = 1
JOB_FINAL = 2

CONTROL_START = 1
CONTROL_SOFT_RESET = 2
STATUS_BUSY = 1
STATUS_DONE = 2
STATUS_ERROR = 4

DMA_CONTROL_RUN = 0x0001
DMA_CONTROL_RESET = 0x0004
DMA_CONTROL_INTERRUPT_ENABLE = 0x1000
DMA_STATUS_HALTED = 0x0001
DMA_STATUS_OFFSET = 0x0004
DMA_RECOVERY_TIMEOUT_SECONDS = 0.1

ERROR_NAMES = {
    0: "NONE",
    1: "INVALID_DIMENSION",
    2: "INVALID_STRIDE",
    3: "BUSY_START",
    4: "STREAM_LENGTH",
    5: "TIMEOUT",
    6: "INVALID_TIMEOUT",
    7: "INVALID_REQUANTIZATION",
}


class NPUError(RuntimeError):
    """Base class for runtime and hardware contract failures."""


class MetadataError(NPUError):
    """Overlay metadata is missing or inconsistent."""


class ABIError(NPUError):
    """The hardware ABI is incompatible with this runtime."""


class ValidationError(NPUError, ValueError):
    """A matrix job violates the public numeric or shape contract."""


class BufferError(NPUError):
    """A DMA allocation is unsafe or inconsistent."""


class DMAError(NPUError):
    """A DMA channel completed with inconsistent transfer metadata."""


class HardwareError(NPUError):
    """The accelerator reported an ABI error code."""

    def __init__(self, code: int, context: str = "matrix job") -> None:
        self.code = int(code)
        self.name = ERROR_NAMES.get(self.code, "UNKNOWN")
        super().__init__(f"{context}: hardware error {self.name} ({self.code})")


@dataclass(frozen=True)
class MatrixJob:
    m: int
    n: int
    k: int
    a_buffer: Any
    b_buffer: Any
    c_buffer: Any


@dataclass(frozen=True)
class MatrixSlice:
    m: int
    n: int
    k: int
    a_buffer: Any
    b_buffer: Any


@dataclass(frozen=True)
class PhysicalJobMetrics:
    cycles: int


class NPURuntime:
    """Submit one physical MxK by KxN signed INT8 matrix job."""

    def __init__(
        self,
        overlay: Any,
        *,
        allocator: Callable[..., Any],
        monotonic: Callable[[], float] = time.monotonic,
        accelerator_name: str = "npu_matrix_accelerator_0",
        dma_name: str = "axi_dma_0",
    ) -> None:
        self.overlay = overlay
        self.allocator = allocator
        self.monotonic = monotonic
        ip_dict = getattr(overlay, "ip_dict", None)
        if not isinstance(ip_dict, dict):
            raise MetadataError("overlay does not expose ip_dict")
        if accelerator_name not in ip_dict or dma_name not in ip_dict:
            raise MetadataError("required accelerator or DMA metadata is missing")

        accelerator_metadata = ip_dict[accelerator_name]
        dma_metadata = ip_dict[dma_name]
        self._validate_address_metadata(accelerator_metadata, accelerator_name)
        self._validate_address_metadata(dma_metadata, dma_name)
        parameters = accelerator_metadata.get("parameters", {})
        try:
            self.max_m = self._positive_parameter(parameters, "ROWS")
            self.max_n = self._positive_parameter(parameters, "COLUMNS")
            self.max_k = self._positive_parameter(parameters, "MAX_K")
        except (TypeError, ValueError, KeyError) as error:
            raise MetadataError("accelerator physical-limit parameters are missing") from error

        try:
            self.mmio = getattr(overlay, accelerator_name)
            self.dma = getattr(overlay, dma_name)
            self.send_channel = self.dma.sendchannel
            self.recv_channel = self.dma.recvchannel
        except AttributeError as error:
            raise MetadataError("overlay IP objects or DMA channels are missing") from error
        self._negotiate_abi()
        self.last_metrics: PhysicalJobMetrics | None = None

    @staticmethod
    def _positive_parameter(parameters: dict[str, Any], name: str) -> int:
        value = int(parameters[name], 0) if isinstance(parameters[name], str) else int(parameters[name])
        if value <= 0:
            raise ValueError(name)
        return value

    @staticmethod
    def _validate_address_metadata(metadata: dict[str, Any], name: str) -> None:
        address = metadata.get("phys_addr", metadata.get("base_address"))
        address_range = metadata.get("addr_range", metadata.get("range"))
        if address is None or address_range is None or int(address_range) <= 0:
            raise MetadataError(f"{name} has incomplete address metadata")

    def _negotiate_abi(self) -> None:
        magic = int(self.mmio.read(REG_MAGIC))
        version = int(self.mmio.read(REG_VERSION))
        capabilities = int(self.mmio.read(REG_CAPABILITIES))
        major = (version >> 16) & 0xFFFF
        if magic != MAGIC:
            raise ABIError(f"bad ABI magic 0x{magic:08x}")
        if major != VERSION_MAJOR:
            raise ABIError(f"unsupported ABI major {major}")
        if capabilities & REQUIRED_CAPABILITIES != REQUIRED_CAPABILITIES:
            raise ABIError(f"missing capabilities 0x{REQUIRED_CAPABILITIES & ~capabilities:08x}")
        self.abi_major = major
        self.capabilities = capabilities

    def _validate_matrix_pair(
        self, a_matrix: np.ndarray, b_matrix: np.ndarray
    ) -> tuple[int, int, int]:
        if not isinstance(a_matrix, np.ndarray) or not isinstance(b_matrix, np.ndarray):
            raise ValidationError("A and B must be NumPy arrays")
        if a_matrix.dtype != np.int8 or b_matrix.dtype != np.int8:
            raise ValidationError("A and B must use signed INT8 dtype")
        if a_matrix.ndim != 2 or b_matrix.ndim != 2:
            raise ValidationError("A and B must be rank-two matrices")
        if not a_matrix.flags.c_contiguous or not b_matrix.flags.c_contiguous:
            raise ValidationError("A and B must be dense C-contiguous matrices")
        m, k = map(int, a_matrix.shape)
        b_k, n = map(int, b_matrix.shape)
        if k != b_k:
            raise ValidationError("A columns must equal B rows")
        if not (1 <= m <= self.max_m and 1 <= n <= self.max_n and 1 <= k <= self.max_k):
            raise ValidationError("matrix shape exceeds physical implementation limits")

        return m, n, k

    def preflight(self, a_matrix: np.ndarray, b_matrix: np.ndarray) -> MatrixJob:
        """Validate one complete quantized job and allocate INT8 output."""

        m, n, k = self._validate_matrix_pair(a_matrix, b_matrix)
        a_buffer = self.allocator(shape=(m, k), dtype=np.int8)
        b_buffer = self.allocator(shape=(k, n), dtype=np.int8)
        c_buffer = self.allocator(shape=(m, n), dtype=np.int8)
        a_buffer[:] = a_matrix
        b_buffer[:] = b_matrix
        self._validate_buffers((a_buffer, b_buffer, c_buffer))
        return MatrixJob(m, n, k, a_buffer, b_buffer, c_buffer)

    @staticmethod
    def _validate_buffers(buffers: tuple[Any, ...]) -> None:
        ranges: list[tuple[int, int]] = []
        for buffer in buffers:
            try:
                address = int(buffer.physical_address)
                size = int(buffer.nbytes)
            except (AttributeError, TypeError, ValueError) as error:
                raise BufferError("DMA buffer lacks physical address or size") from error
            end = address + size
            if address < 0 or address % 64 != 0 or size <= 0 or end > (1 << 32):
                raise BufferError("DMA buffer is unaligned, empty, or wraps its physical range")
            ranges.append((address, end))
        for left_index, left in enumerate(ranges):
            for right in ranges[left_index + 1 :]:
                if left[0] < right[1] and right[0] < left[1]:
                    raise BufferError("DMA buffers overlap")

    @staticmethod
    def _validate_quantization(
        bias: np.ndarray,
        multipliers_q31: np.ndarray,
        shifts: np.ndarray,
        output_zero_point: int,
        channels: int,
    ) -> None:
        if (
            not isinstance(bias, np.ndarray)
            or bias.dtype != np.int32
            or bias.shape != (channels,)
        ):
            raise ValidationError(
                f"bias must be signed INT32 with shape ({channels},)"
            )
        if (
            not isinstance(multipliers_q31, np.ndarray)
            or multipliers_q31.dtype != np.int32
            or multipliers_q31.shape != (channels,)
        ):
            raise ValidationError(
                f"multipliers_q31 must be signed INT32 with shape ({channels},)"
            )
        if (
            not isinstance(shifts, np.ndarray)
            or shifts.dtype != np.uint8
            or shifts.shape != (channels,)
        ):
            raise ValidationError(
                f"shifts must be unsigned INT8 with shape ({channels},)"
            )
        if np.any(shifts > 31):
            raise ValidationError("shifts must be in [0, 31]")
        if (
            isinstance(output_zero_point, bool)
            or not isinstance(output_zero_point, (int, np.integer))
            or not -128 <= int(output_zero_point) <= 127
        ):
            raise ValidationError("output_zero_point must be a signed INT8 integer")

    def run(
        self,
        a_matrix: np.ndarray,
        b_matrix: np.ndarray,
        *,
        bias: np.ndarray,
        multipliers_q31: np.ndarray,
        shifts: np.ndarray,
        output_zero_point: int,
        hardware_timeout_cycles: int = 1_000_000,
        software_timeout: float = 5.0,
    ) -> np.ndarray:
        return self.run_slices(
            (a_matrix,),
            (b_matrix,),
            bias=bias,
            multipliers_q31=multipliers_q31,
            shifts=shifts,
            output_zero_point=output_zero_point,
            hardware_timeout_cycles=hardware_timeout_cycles,
            software_timeout=software_timeout,
        )

    def run_slices(
        self,
        a_tiles: tuple[np.ndarray, ...] | list[np.ndarray],
        b_tiles: tuple[np.ndarray, ...] | list[np.ndarray],
        *,
        bias: np.ndarray,
        multipliers_q31: np.ndarray,
        shifts: np.ndarray,
        output_zero_point: int,
        hardware_timeout_cycles: int = 1_000_000,
        software_timeout: float = 5.0,
    ) -> np.ndarray:
        """Accumulate K slices and return the hardware-produced INT8 tile."""

        self.last_metrics = None
        if not isinstance(hardware_timeout_cycles, int) or hardware_timeout_cycles <= 0:
            raise ValidationError("hardware timeout must be a positive integer")
        if software_timeout <= 0:
            raise ValidationError("software timeout must be positive")
        try:
            a_tiles = tuple(a_tiles)
            b_tiles = tuple(b_tiles)
        except TypeError as error:
            raise ValidationError("matrix slices must be sequences") from error
        if not a_tiles or len(a_tiles) != len(b_tiles):
            raise ValidationError("A and B slice sequences must have equal nonzero length")

        slices: list[MatrixSlice] = []
        buffers: list[Any] = []
        expected_m = expected_n = None
        for a_matrix, b_matrix in zip(a_tiles, b_tiles):
            m, n, k = self._validate_matrix_pair(a_matrix, b_matrix)
            if expected_m is None:
                expected_m, expected_n = m, n
            elif (m, n) != (expected_m, expected_n):
                raise ValidationError("all K slices must use identical M and N")
            a_buffer = self.allocator(shape=(m, k), dtype=np.int8)
            b_buffer = self.allocator(shape=(k, n), dtype=np.int8)
            a_buffer[:] = a_matrix
            b_buffer[:] = b_matrix
            slices.append(MatrixSlice(m, n, k, a_buffer, b_buffer))
            buffers.extend((a_buffer, b_buffer))

        assert expected_m is not None and expected_n is not None
        self._validate_quantization(
            bias, multipliers_q31, shifts, output_zero_point, expected_n
        )
        c_buffer = self.allocator(shape=(expected_m, expected_n), dtype=np.int8)
        bias_buffer = self.allocator(shape=(expected_n,), dtype=np.int32)
        multiplier_buffer = self.allocator(shape=(expected_n,), dtype=np.int32)
        shift_buffer = self.allocator(shape=(expected_n,), dtype=np.uint8)
        bias_buffer[:] = bias
        multiplier_buffer[:] = multipliers_q31
        shift_buffer[:] = shifts
        buffers.extend((c_buffer, bias_buffer, multiplier_buffer, shift_buffer))
        self._validate_buffers(tuple(buffers))

        deadline = self.monotonic() + float(software_timeout)
        cycle_total = 0
        try:
            self.mmio.write(REG_CONTROL, CONTROL_SOFT_RESET)
            for index, job in enumerate(slices):
                final = index == len(slices) - 1
                flags = (JOB_FIRST if index == 0 else 0) | (
                    JOB_FINAL if final else 0
                )
                self.mmio.write(REG_M, job.m)
                self.mmio.write(REG_N, job.n)
                self.mmio.write(REG_K, job.k)
                self.mmio.write(REG_A_STRIDE, job.k)
                self.mmio.write(REG_B_STRIDE, job.n)
                self.mmio.write(REG_C_STRIDE, job.n)
                self.mmio.write(REG_TIMEOUT_CYCLES, hardware_timeout_cycles)
                self.mmio.write(REG_JOB_FLAGS, flags)
                self.mmio.write(REG_OUTPUT_ZERO_POINT, int(output_zero_point) & 0xFF)

                if final:
                    self.recv_channel.transfer(
                        c_buffer, nbytes=job.m * job.n
                    )
                self.mmio.write(REG_CONTROL, CONTROL_START)
                job.a_buffer.flush()
                self.send_channel.transfer(job.a_buffer, nbytes=job.m * job.k)
                self._wait_channel(self.send_channel, deadline, "A MM2S")
                self._check_length(self.send_channel, job.m * job.k, "A MM2S")
                job.b_buffer.flush()
                self.send_channel.transfer(job.b_buffer, nbytes=job.k * job.n)
                self._wait_channel(self.send_channel, deadline, "B MM2S")
                self._check_length(self.send_channel, job.k * job.n, "B MM2S")

                if final:
                    for parameter_buffer, byte_count, label in (
                        (bias_buffer, 4 * job.n, "bias MM2S"),
                        (multiplier_buffer, 4 * job.n, "multiplier MM2S"),
                        (shift_buffer, job.n, "shift MM2S"),
                    ):
                        parameter_buffer.flush()
                        self.send_channel.transfer(parameter_buffer, nbytes=byte_count)
                        self._wait_channel(self.send_channel, deadline, label)
                        self._check_length(self.send_channel, byte_count, label)

                status = self._wait_hardware(deadline)
                if status & STATUS_ERROR:
                    raise HardwareError(int(self.mmio.read(REG_ERROR)))
                if not status & STATUS_DONE:
                    raise HardwareError(
                        int(self.mmio.read(REG_ERROR)),
                        "DONE was not asserted",
                    )
                cycle_total += self._read_cycles(deadline)

            self._wait_channel(self.recv_channel, deadline, "C S2MM")
            self._check_length(
                self.recv_channel, expected_m * expected_n, "C S2MM"
            )
            c_buffer.invalidate()
            result = np.array(c_buffer, copy=True)
            if result.dtype != np.int8 or result.shape != (expected_m, expected_n):
                raise DMAError(
                    "hardware output buffer is not signed INT8 with the expected shape"
                )
            self.last_metrics = PhysicalJobMetrics(cycles=cycle_total)
            return result
        except Exception:
            self._recover()
            raise

    def _wait_channel(self, channel: Any, deadline: float, label: str) -> None:
        while not bool(getattr(channel, "idle", False)):
            if self.monotonic() >= deadline:
                raise TimeoutError(f"{label} timed out")
        wait = getattr(channel, "wait", None)
        if callable(wait):
            # PYNQ 3.1 finalizes error, cache, and transferred-byte state here.
            # The idle poll above makes this driver call nonblocking.
            wait()

    @staticmethod
    def _check_length(channel: Any, expected: int, label: str) -> None:
        actual = getattr(channel, "transferred", expected)
        if actual is not None and int(actual) != expected:
            raise DMAError(f"{label} transferred {actual} bytes, expected {expected}")

    def _wait_hardware(self, deadline: float) -> int:
        while True:
            status = int(self.mmio.read(REG_STATUS))
            if status & STATUS_ERROR or status & STATUS_DONE or not status & STATUS_BUSY:
                return status
            if self.monotonic() >= deadline:
                raise TimeoutError("accelerator status timed out")

    def _read_cycles(self, deadline: float) -> int:
        while True:
            if self.monotonic() >= deadline:
                raise TimeoutError("cycle counter read timed out")
            high_before = int(self.mmio.read(REG_CYCLES_HI))
            low = int(self.mmio.read(REG_CYCLES_LO))
            high_after = int(self.mmio.read(REG_CYCLES_HI))
            for value in (high_before, low, high_after):
                if not 0 <= value <= 0xFFFFFFFF:
                    raise HardwareError(0, "cycle counter word is invalid")
            if high_before == high_after:
                return (high_before << 32) | low

    def _recover(self) -> None:
        try:
            self.mmio.write(REG_CONTROL, CONTROL_SOFT_RESET)
        except Exception:
            pass
        deadline = self.monotonic() + DMA_RECOVERY_TIMEOUT_SECONDS
        if not self._reset_dma(self.send_channel, deadline):
            return
        for channel in (self.send_channel, self.recv_channel):
            self._restart_dma_channel(channel, deadline)

    def _dma_access(self, channel: Any) -> tuple[Any, int] | None:
        dma_mmio = getattr(channel, "_mmio", None)
        offset = getattr(channel, "_offset", None)
        read = getattr(dma_mmio, "read", None)
        write = getattr(dma_mmio, "write", None)
        if not callable(read) or not callable(write) or not isinstance(offset, int):
            return None
        return dma_mmio, offset

    def _reset_dma(self, channel: Any, deadline: float) -> bool:
        """Reset the shared AXI DMA core once, with a bounded poll."""

        access = self._dma_access(channel)
        if access is None:
            return False
        dma_mmio, offset = access
        try:
            dma_mmio.write(offset, DMA_CONTROL_RESET)
            while int(dma_mmio.read(offset)) & DMA_CONTROL_RESET:
                if self.monotonic() >= deadline:
                    return False
            return True
        except Exception:
            return False

    def _restart_dma_channel(self, channel: Any, deadline: float) -> bool:
        """Restart one channel without PYNQ's unbounded start loop."""

        access = self._dma_access(channel)
        if access is None:
            return False
        dma_mmio, offset = access
        try:
            run_value = DMA_CONTROL_RUN
            if bool(getattr(channel, "_interrupt", False)):
                run_value |= DMA_CONTROL_INTERRUPT_ENABLE
            dma_mmio.write(offset, run_value)
            while int(dma_mmio.read(offset + DMA_STATUS_OFFSET)) & DMA_STATUS_HALTED:
                if self.monotonic() >= deadline:
                    return False
            if hasattr(channel, "_first_transfer"):
                channel._first_transfer = True
            return True
        except Exception:
            return False


def load_pynq_runtime(bitstream: str | Path, **runtime_kwargs: Any) -> NPURuntime:
    """Load a matching BIT/HWH pair; importing this module never imports PYNQ."""

    bit_path = Path(bitstream).expanduser().resolve()
    hwh_path = bit_path.with_suffix(".hwh")
    if bit_path.suffix.lower() != ".bit" or not bit_path.is_file() or not hwh_path.is_file():
        raise MetadataError("a same-basename .bit/.hwh overlay pair is required")
    from pynq import Overlay, allocate  # type: ignore[import-not-found]

    overlay = Overlay(str(bit_path), download=True)
    return NPURuntime(overlay, allocator=allocate, **runtime_kwargs)
