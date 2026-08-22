"""Bounded Phase 1B matrix runtime for a DMA-connected PYNQ overlay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np


MAGIC = 0x3155504E
VERSION_MAJOR = 1
REQUIRED_CAPABILITIES = 0x1B

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

CONTROL_START = 1
CONTROL_SOFT_RESET = 2
STATUS_BUSY = 1
STATUS_DONE = 2
STATUS_ERROR = 4

ERROR_NAMES = {
    0: "NONE",
    1: "INVALID_DIMENSION",
    2: "INVALID_STRIDE",
    3: "BUSY_START",
    4: "STREAM_LENGTH",
    5: "TIMEOUT",
    6: "INVALID_TIMEOUT",
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

    def preflight(self, a_matrix: np.ndarray, b_matrix: np.ndarray) -> MatrixJob:
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

        a_buffer = self.allocator(shape=(m, k), dtype=np.int8)
        b_buffer = self.allocator(shape=(k, n), dtype=np.int8)
        c_buffer = self.allocator(shape=(m, n), dtype=np.int32)
        a_buffer[:] = a_matrix
        b_buffer[:] = b_matrix
        self._validate_buffers((a_buffer, b_buffer, c_buffer))
        return MatrixJob(m, n, k, a_buffer, b_buffer, c_buffer)

    @staticmethod
    def _validate_buffers(buffers: tuple[Any, Any, Any]) -> None:
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

    def run(
        self,
        a_matrix: np.ndarray,
        b_matrix: np.ndarray,
        *,
        hardware_timeout_cycles: int = 1_000_000,
        software_timeout: float = 5.0,
    ) -> np.ndarray:
        if not isinstance(hardware_timeout_cycles, int) or hardware_timeout_cycles <= 0:
            raise ValidationError("hardware timeout must be a positive integer")
        if software_timeout <= 0:
            raise ValidationError("software timeout must be positive")
        job = self.preflight(a_matrix, b_matrix)
        deadline = self.monotonic() + float(software_timeout)
        try:
            self.mmio.write(REG_CONTROL, CONTROL_SOFT_RESET)
            self.mmio.write(REG_M, job.m)
            self.mmio.write(REG_N, job.n)
            self.mmio.write(REG_K, job.k)
            self.mmio.write(REG_A_STRIDE, job.k)
            self.mmio.write(REG_B_STRIDE, job.n)
            self.mmio.write(REG_C_STRIDE, 4 * job.n)
            self.mmio.write(REG_TIMEOUT_CYCLES, hardware_timeout_cycles)

            self.recv_channel.transfer(job.c_buffer, nbytes=4 * job.m * job.n)
            self.mmio.write(REG_CONTROL, CONTROL_START)
            job.a_buffer.flush()
            self.send_channel.transfer(job.a_buffer, nbytes=job.m * job.k)
            self._wait_channel(self.send_channel, deadline, "A MM2S")
            self._check_length(self.send_channel, job.m * job.k, "A MM2S")
            job.b_buffer.flush()
            self.send_channel.transfer(job.b_buffer, nbytes=job.k * job.n)
            self._wait_channel(self.send_channel, deadline, "B MM2S")
            self._check_length(self.send_channel, job.k * job.n, "B MM2S")

            status = self._wait_hardware(deadline)
            self._wait_channel(self.recv_channel, deadline, "C S2MM")
            self._check_length(self.recv_channel, 4 * job.m * job.n, "C S2MM")
            if status & STATUS_ERROR:
                raise HardwareError(int(self.mmio.read(REG_ERROR)))
            if not status & STATUS_DONE:
                raise HardwareError(int(self.mmio.read(REG_ERROR)), "DONE was not asserted")
            job.c_buffer.invalidate()
            return np.array(np.asarray(job.c_buffer), dtype=np.int32, copy=True).reshape(job.m, job.n)
        except Exception:
            self._recover()
            raise

    def _wait_channel(self, channel: Any, deadline: float, label: str) -> None:
        while not bool(getattr(channel, "idle", False)):
            if self.monotonic() >= deadline:
                raise TimeoutError(f"{label} timed out")

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

    def _recover(self) -> None:
        try:
            self.mmio.write(REG_CONTROL, CONTROL_SOFT_RESET)
        except Exception:
            pass
        for channel in (self.send_channel, self.recv_channel):
            stop = getattr(channel, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass


def load_pynq_runtime(bitstream: str | Path, **runtime_kwargs: Any) -> NPURuntime:
    """Load a matching BIT/HWH pair; importing this module never imports PYNQ."""

    bit_path = Path(bitstream).expanduser().resolve()
    hwh_path = bit_path.with_suffix(".hwh")
    if bit_path.suffix.lower() != ".bit" or not bit_path.is_file() or not hwh_path.is_file():
        raise MetadataError("a same-basename .bit/.hwh overlay pair is required")
    from pynq import Overlay, allocate  # type: ignore[import-not-found]

    overlay = Overlay(str(bit_path), download=True)
    return NPURuntime(overlay, allocator=allocate, **runtime_kwargs)
