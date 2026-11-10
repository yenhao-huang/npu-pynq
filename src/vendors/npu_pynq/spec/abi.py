"""The NPU's software-visible interface (ABI 2.0): how other code talks to it.

``isa.py`` is what the NPU reads in its instruction stream; this is everything
around it: identity and version, capability bits, the AXI-Lite register map,
control and status bits, and error codes. The RTL (``csr/``, the control and status registers) and the board
driver (``driver/npu_rt/pynq.cpp``) must match it; ``src/test/tests/test_abi.py``
and ``test_isa.py`` check both.

Versioning: the minor version grows when registers or capability bits are
added (ABI 1.1 added the instruction-stream front end, ``ISA_FRONTEND``); the
major version grows when software written for the previous one can no longer
work, and ``negotiate_abi`` rejects a different major.

ABI 2.0 retires the ABI 1.x direct-job path. The CPU starts programs through
``IsaRegister``; the scalar core is the matrix core's only command source.
``CONTROL`` bit 0 (``START``) is ignored, and the job registers at 0x18-0x30
(``RETIRED_REGISTERS``) are reserved: they read as zero and ignore writes.
Since the Xilinx AXI DMA was removed (#150) a direct job had no operand
source, and the board driver never used it. The capability word is unchanged.
"""

from dataclasses import dataclass
from enum import IntEnum, IntFlag

ABI_MAGIC = 0x3155504E
ABI_MAJOR = 2
ABI_MINOR = 0
ABI_WINDOW_BYTES = 0x100
PHYSICAL_ADDRESS_LIMIT = 1 << 32
MAX_DIMENSION = 0xFFFF


class Capability(IntFlag):
    MATRIX_INT8 = 1 << 0
    SATURATING_ACCUM_INT32 = 1 << 1
    REQUANT_INT8 = 1 << 2
    STREAM_TLAST = 1 << 3
    CYCLE_COUNTER = 1 << 4
    #: The matrix core queues two jobs (STATUS.ACCEPT); the scalar core
    #: issues the next GEMM while the current one runs.
    PIPELINED_JOBS = 1 << 5
    #: ABI 1.1: the NPU fetches and decodes an instruction stream (isa.py)
    #: from DDR through its own AXI4 master; see IsaRegister. Since ABI 2.0 it
    #: is the only way to run a job.
    ISA_FRONTEND = 1 << 6


#: What the runtime needs: INT8 GEMM with saturating INT32 results through the
#: instruction-stream front end. PIPELINED_JOBS is advertised, not required.
#: STREAM_TLAST described the AXI-Stream input of the removed DMA path (#150);
#: the hardware still reports the bit, but nothing requires it.
REQUIRED_CAPABILITIES = (
    Capability.MATRIX_INT8
    | Capability.SATURATING_ACCUM_INT32
    | Capability.CYCLE_COUNTER
    | Capability.ISA_FRONTEND
)


class Register(IntEnum):
    """Identity and matrix-core status. STATUS, ERROR and CYCLES report the
    matrix core's most recent jobs, which the scalar core issued."""

    MAGIC = 0x00
    VERSION = 0x04
    CAPABILITIES = 0x08
    CONTROL = 0x0C
    STATUS = 0x10
    ERROR = 0x14
    CYCLES_LO = 0x34
    CYCLES_HI = 0x38


#: ABI 1.x direct-job registers retired in ABI 2.0. The offsets are reserved:
#: they read as zero and ignore writes. Strides and the timeout are not
#: configurable; the scalar core always passes dense row-major strides and the
#: largest timeout.
RETIRED_REGISTERS = {
    "M": 0x18,
    "N": 0x1C,
    "K": 0x20,
    "A_STRIDE": 0x24,
    "B_STRIDE": 0x28,
    "C_STRIDE": 0x2C,
    "TIMEOUT_CYCLES": 0x30,
}


class IsaRegister(IntEnum):
    """Instruction-stream front end registers (ABI 1.1, ISA_FRONTEND)."""

    CONTROL = 0x40
    STATUS = 0x44
    PROG_ADDR = 0x48
    PROG_LEN = 0x4C
    DATA_BASE = 0x50
    ERROR = 0x54
    PC = 0x58
    JOBS = 0x5C
    CYCLES = 0x60
    INSTRUCTIONS = 0x64
    VERSION = 0x68


class IsaStatus(IntFlag):
    RUNNING = 1 << 0
    DONE = 1 << 1
    ERROR = 1 << 2
    BUSY = 1 << 3


class Control(IntFlag):
    """CONTROL bits. Bit 0 was START, retired in ABI 2.0 and ignored."""

    SOFT_RESET = 1 << 1


#: CONTROL bit 0 (ABI 1.x START), ignored since ABI 2.0.
RETIRED_CONTROL_BITS = 1 << 0


class Status(IntFlag):
    BUSY = 1 << 0
    DONE = 1 << 1
    ERROR = 1 << 2
    #: The matrix core's job queue has a free entry, so the scalar core may
    #: issue the next job. Devices without Capability.PIPELINED_JOBS never
    #: report it.
    ACCEPT = 1 << 3


class ErrorCode(IntEnum):
    """Matrix-core error codes (ERROR, and forwarded in ISA_ERROR).

    INVALID_STRIDE and INVALID_TIMEOUT remain the controller's codes, but the
    scalar core always passes dense strides and the largest timeout, so ABI
    2.0 hardware does not raise them.
    """

    NONE = 0
    INVALID_DIMENSION = 1
    INVALID_STRIDE = 2
    BUSY_START = 3
    STREAM_LENGTH = 4
    TIMEOUT = 5
    INVALID_TIMEOUT = 6
    INTERNAL = 255


class AbiCompatibilityError(ValueError):
    """Raised when a device cannot satisfy this software ABI."""


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
    required: Capability = REQUIRED_CAPABILITIES,
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
