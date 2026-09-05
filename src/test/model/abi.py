"""Version 2 software-visible hardware ABI for the PYNQ NPU."""

from dataclasses import dataclass
from enum import IntEnum, IntFlag

ABI_MAGIC = 0x3155504E
ABI_MAJOR = 2
ABI_MINOR = 0
ABI_WINDOW_BYTES = 0x100
DMA_ALIGNMENT_BYTES = 64
PHYSICAL_ADDRESS_LIMIT = 1 << 32
MAX_DIMENSION = 0xFFFF


class Capability(IntFlag):
    MATRIX_INT8 = 1 << 0
    SATURATING_ACCUM_INT32 = 1 << 1
    REQUANT_INT8 = 1 << 2
    STREAM_TLAST = 1 << 3
    CYCLE_COUNTER = 1 << 4


MATRIX_REQUIRED_CAPABILITIES = (
    Capability.MATRIX_INT8
    | Capability.SATURATING_ACCUM_INT32
    | Capability.REQUANT_INT8
    | Capability.STREAM_TLAST
    | Capability.CYCLE_COUNTER
)


class Register(IntEnum):
    MAGIC = 0x00
    VERSION = 0x04
    CAPABILITIES = 0x08
    CONTROL = 0x0C
    STATUS = 0x10
    ERROR = 0x14
    M = 0x18
    N = 0x1C
    K = 0x20
    A_STRIDE = 0x24
    B_STRIDE = 0x28
    C_STRIDE = 0x2C
    TIMEOUT_CYCLES = 0x30
    CYCLES_LO = 0x34
    CYCLES_HI = 0x38
    JOB_FLAGS = 0x3C
    OUTPUT_ZERO_POINT = 0x40


class Control(IntFlag):
    START = 1 << 0
    SOFT_RESET = 1 << 1


class Status(IntFlag):
    BUSY = 1 << 0
    DONE = 1 << 1
    ERROR = 1 << 2


class ErrorCode(IntEnum):
    NONE = 0
    INVALID_DIMENSION = 1
    INVALID_STRIDE = 2
    BUSY_START = 3
    STREAM_LENGTH = 4
    TIMEOUT = 5
    INVALID_TIMEOUT = 6
    INVALID_REQUANTIZATION = 7
    INTERNAL = 255


class AbiCompatibilityError(ValueError):
    """Raised when a device cannot satisfy this software ABI."""


class JobValidationError(ValueError):
    """Software preflight failure with the matching hardware ABI error code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _bounded(name: str, value: int, minimum: int, maximum: int) -> int:
    value = _integer(name, value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


@dataclass(frozen=True)
class AbiVersion:
    major: int = ABI_MAJOR
    minor: int = ABI_MINOR

    def __post_init__(self) -> None:
        _bounded("major", self.major, 0, 0xFFFF)
        _bounded("minor", self.minor, 0, 0xFFFF)

    def encode(self) -> int:
        return (self.major << 16) | self.minor

    @classmethod
    def decode(cls, word: int) -> "AbiVersion":
        word = _bounded("version_word", word, 0, 0xFFFFFFFF)
        return cls(major=(word >> 16) & 0xFFFF, minor=word & 0xFFFF)


def negotiate_abi(
    magic: int,
    version_word: int,
    capabilities: int,
    required: Capability = MATRIX_REQUIRED_CAPABILITIES,
) -> AbiVersion:
    """Validate identity, major version, and required advertised capabilities."""

    if _bounded("magic", magic, 0, 0xFFFFFFFF) != ABI_MAGIC:
        raise AbiCompatibilityError(f"unexpected ABI magic 0x{magic:08X}")
    version = AbiVersion.decode(version_word)
    if version.major != ABI_MAJOR:
        raise AbiCompatibilityError(
            f"unsupported ABI major {version.major}; expected {ABI_MAJOR}"
        )
    reported = Capability(_bounded("capabilities", capabilities, 0, 0xFFFFFFFF))
    required = Capability(required)
    missing = required & ~reported
    if missing:
        raise AbiCompatibilityError(f"missing required capabilities: {missing!s}")
    return version


@dataclass(frozen=True)
class MatrixJob:
    m: int
    n: int
    k: int
    a_stride: int
    b_stride: int
    c_stride: int
    timeout_cycles: int

    def __post_init__(self) -> None:
        for name in ("m", "n", "k"):
            try:
                _bounded(name, getattr(self, name), 1, MAX_DIMENSION)
            except (TypeError, ValueError) as error:
                raise JobValidationError(ErrorCode.INVALID_DIMENSION, str(error)) from error
        try:
            _bounded("timeout_cycles", self.timeout_cycles, 1, 0xFFFFFFFF)
        except (TypeError, ValueError) as error:
            raise JobValidationError(ErrorCode.INVALID_TIMEOUT, str(error)) from error
        self._validate_stride("a_stride", self.a_stride, self.k, 1)
        self._validate_stride("b_stride", self.b_stride, self.n, 1)
        self._validate_stride("c_stride", self.c_stride, self.n, 1)

    @staticmethod
    def _validate_stride(
        name: str, value: int, minimum_bytes: int, element_bytes: int
    ) -> None:
        try:
            value = _bounded(name, value, 1, 0xFFFFFFFF)
        except (TypeError, ValueError) as error:
            raise JobValidationError(ErrorCode.INVALID_STRIDE, str(error)) from error
        if value < minimum_bytes or value % element_bytes:
            raise JobValidationError(
                ErrorCode.INVALID_STRIDE,
                f"{name} must cover {minimum_bytes} bytes and be a multiple "
                f"of {element_bytes}, got {value}"
            )

    @classmethod
    def dense(
        cls, m: int, n: int, k: int, timeout_cycles: int
    ) -> "MatrixJob":
        return cls(
            m=m,
            n=n,
            k=k,
            a_stride=k,
            b_stride=n,
            c_stride=n,
            timeout_cycles=timeout_cycles,
        )

    @property
    def input_elements(self) -> int:
        return self.m * self.k + self.k * self.n

    @property
    def output_elements(self) -> int:
        return self.m * self.n

    @property
    def payload_bytes(self) -> tuple[int, int, int]:
        return self.m * self.k, self.k * self.n, self.m * self.n


@dataclass(frozen=True)
class BufferRange:
    address: int
    size: int

    def __post_init__(self) -> None:
        address = _bounded("address", self.address, 0, PHYSICAL_ADDRESS_LIMIT - 1)
        size = _bounded("size", self.size, 1, PHYSICAL_ADDRESS_LIMIT)
        if address % DMA_ALIGNMENT_BYTES:
            raise ValueError(
                f"address must be {DMA_ALIGNMENT_BYTES}-byte aligned, "
                f"got 0x{address:X}"
            )
        if address + size > PHYSICAL_ADDRESS_LIMIT:
            raise ValueError("buffer range wraps beyond the 32-bit address space")

    @property
    def end(self) -> int:
        return self.address + self.size

    def overlaps(self, other: "BufferRange") -> bool:
        return self.address < other.end and other.address < self.end


@dataclass(frozen=True)
class MatrixBuffers:
    a: BufferRange
    b: BufferRange
    c: BufferRange

    def validate_for(self, job: MatrixJob) -> None:
        required_sizes = job.payload_bytes
        for name, buffer_range, required in zip(
            ("a", "b", "c"), (self.a, self.b, self.c), required_sizes
        ):
            if buffer_range.size < required:
                raise ValueError(
                    f"{name} buffer requires {required} bytes, "
                    f"got {buffer_range.size}"
                )
        if self.c.overlaps(self.a) or self.c.overlaps(self.b):
            raise ValueError("writable c buffer must not overlap input buffers")
