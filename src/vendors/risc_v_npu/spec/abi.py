"""The CVA6 NPU's software-visible interface: how other code talks to it.

Two windows, seen by two processors:

- The **CSR window** (AXI-Lite, at the overlay's ``C_BASEADDR``) is what the
  board's ARM CPU uses. It is the npu_pynq ABI 2.0 unchanged (identity,
  capabilities, matrix-core status, the ``IsaRegister`` block that starts a
  program and reports its end) plus one capability bit,
  ``RISCV_SCALAR_CORE``, set only by the ``SCALAR_IMPL = "CVA6"`` build. On
  this build ``PROG_ADDR`` points at a RISC-V binary instead of ISA words, and
  ``PROG_LEN`` is that binary's size in 64-bit words, rounded up.
- The **MMIO window** (``MMIO_BASE``, 4 KiB) is what the program running on
  CVA6 uses to issue GEMM jobs to the matrix core and to end itself. It is not
  visible to the ARM CPU. ``src/hw/rtl/npu_accelerator/scalar_core/cva6/
  cva6_npu_mmio.sv`` implements it; ``src/test/tests/test_risc_v_npu_abi.py``
  checks the two agree, and ``gen_mmio_header.py`` turns this file into the C
  header the compiler and runtime code include.

The CSR part is copied from src/vendors/npu_pynq/spec/abi.py (risc_v_npu
keeps its own copy; plan 2026-1112-003 section 6.2). Keep the two in step.
"""

from dataclasses import dataclass
from enum import IntEnum, IntFlag

# ----------------------------------------------------------------- CSR window

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
    PIPELINED_JOBS = 1 << 5
    ISA_FRONTEND = 1 << 6
    #: The scalar core is CVA6 (SCALAR_IMPL = "CVA6"): PROG_ADDR points at an
    #: RV32IMC binary and the program drives the matrix core through the MMIO
    #: window below. Not set by the default (CUSTOM) build.
    RISCV_SCALAR_CORE = 1 << 7


#: What librv_npu_rt needs: the npu_pynq requirements plus a CVA6 scalar core.
#: ISA_FRONTEND is kept: it is the IsaRegister block that starts a program.
REQUIRED_CAPABILITIES = (
    Capability.MATRIX_INT8
    | Capability.SATURATING_ACCUM_INT32
    | Capability.CYCLE_COUNTER
    | Capability.ISA_FRONTEND
    | Capability.RISCV_SCALAR_CORE
)


class Register(IntEnum):
    """Identity and matrix-core status (as npu_pynq)."""

    MAGIC = 0x00
    VERSION = 0x04
    CAPABILITIES = 0x08
    CONTROL = 0x0C
    STATUS = 0x10
    ERROR = 0x14
    CYCLES_LO = 0x34
    CYCLES_HI = 0x38


#: ABI 1.x direct-job registers retired in ABI 2.0 (as npu_pynq).
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
    """Program start and end (as npu_pynq). On the CVA6 build PROG_ADDR is the
    RISC-V binary's address (CVA6's boot address), PROG_LEN its size in 64-bit
    words (rounded up), DATA_BASE the delegate's data block; the program reads
    the last two back through the MMIO window. JOBS and INSTRUCTIONS count NPU
    jobs; PC reports the boot address (plan 2026-1112-003 section 9.2)."""

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


RETIRED_CONTROL_BITS = 1 << 0


class Status(IntFlag):
    BUSY = 1 << 0
    DONE = 1 << 1
    ERROR = 1 << 2
    ACCEPT = 1 << 3


class ErrorCode(IntEnum):
    """Matrix-core error codes (ERROR, and forwarded in ISA_ERROR)."""

    NONE = 0
    INVALID_DIMENSION = 1
    INVALID_STRIDE = 2
    BUSY_START = 3
    STREAM_LENGTH = 4
    TIMEOUT = 5
    INVALID_TIMEOUT = 6
    INTERNAL = 255


# ---------------------------------------------------------------- MMIO window

#: Where CVA6 sees the job registers: uncached, non-idempotent, never on m_axi.
MMIO_BASE = 0x4000_0000
MMIO_BYTES = 0x1000

#: CVA6's view of DDR (cached, executable): programs, A, B and C must lie here.
DDR_BASE = 0x0000_0000
DDR_BYTES = 0x2000_0000

#: Anything outside DDR and MMIO is answered DECERR (axi_err_slv).
ADDRESS_MAP = (
    ("ddr", DDR_BASE, DDR_BYTES),
    ("mmio", MMIO_BASE, MMIO_BYTES),
)

#: A, B and C addresses written to the MMIO window must be this aligned.
MMIO_OPERAND_ALIGN = 8


class MmioRegister(IntEnum):
    """32-bit registers, offsets from MMIO_BASE. Single-beat accesses only."""

    SHAPE_MN = 0x00   # M [15:0], N [31:16]
    SHAPE_K = 0x04    # K [15:0]
    ADDR_A = 0x08
    ADDR_B = 0x0C
    ADDR_C = 0x10
    DOORBELL = 0x14   # issue one job; the write response waits while the queue is full
    STATUS = 0x18     # see MmioStatus
    FENCE = 0x1C      # as STATUS, answered once every issued job is written back
    DONE = 0x20       # [7:0] code, 0 = success; ends the program
    DATA_BASE = 0x24  # the CSR data_base of this run
    PROG_LEN = 0x28   # the CSR prog_len of this run


#: Access type per register: R, W or RW. A write to an R register, a read of a
#: W register, an unknown offset or a burst is answered SLVERR.
MMIO_ACCESS = {
    MmioRegister.SHAPE_MN: "RW",
    MmioRegister.SHAPE_K: "RW",
    MmioRegister.ADDR_A: "RW",
    MmioRegister.ADDR_B: "RW",
    MmioRegister.ADDR_C: "RW",
    MmioRegister.DOORBELL: "W",
    MmioRegister.STATUS: "R",
    MmioRegister.FENCE: "R",
    MmioRegister.DONE: "W",
    MmioRegister.DATA_BASE: "R",
    MmioRegister.PROG_LEN: "R",
}


class MmioStatus(IntFlag):
    """Single-bit fields of MMIO STATUS / FENCE."""

    BUSY = 1 << 0     # a job is queued or not yet written back
    ERROR = 1 << 1    # sticky until the next program start


#: Multi-bit fields of MMIO STATUS / FENCE: name -> (shift, width).
MMIO_STATUS_FIELDS = {
    "ERROR_CODE": (8, 8),     # the first fault's code (MmioError or ErrorCode)
    "OUTSTANDING": (16, 8),   # jobs queued or issued and not written back
}

#: Shape fields: name -> (register, shift, width).
MMIO_SHAPE_FIELDS = {
    "M": (MmioRegister.SHAPE_MN, 0, 16),
    "N": (MmioRegister.SHAPE_MN, 16, 16),
    "K": (MmioRegister.SHAPE_K, 0, 16),
}

#: DONE's code field.
MMIO_DONE_CODE = (0, 8)


class MmioError(IntEnum):
    """Faults the MMIO block latches (STATUS.ERROR_CODE, then ISA_ERROR).
    The same values as the CUSTOM core's front-end codes in
    src/vendors/npu_pynq/spec/isa.py (ERR_*); matrix-core faults forward
    ErrorCode instead."""

    BAD_SHAPE = 0x11     # a shape outside 1..ROWS x 1..COLUMNS x 1..MAX_K
    MISALIGNED = 0x12    # A, B or C not 8-byte aligned
    AXI_READ = 0x13      # the tensor DMA's read failed
    AXI_WRITE = 0x14     # the tensor DMA's write failed


class DoneCode(IntEnum):
    """Codes a program writes to DONE. Any nonzero code ends the run with
    ISA_STATUS.ERROR and ISA_ERROR = code; a latched MmioError wins over it."""

    SUCCESS = 0x00
    #: Written by the runtime's trap handler (custom_dialect/runtime/crt0.S)
    #: when the program takes an exception. Proposed here; not in the RTL.
    TRAP = 0x20


# ----------------------------------------------------------------- functions

class AbiCompatibilityError(ValueError):
    """Raised when a device cannot satisfy this software ABI."""


def _bounded(name: str, value: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
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
    """Validate identity, major version and the required capabilities
    (RISCV_SCALAR_CORE among them: a CUSTOM overlay is refused)."""

    if _bounded("magic", magic, 0, 0xFFFFFFFF) != ABI_MAGIC:
        raise AbiCompatibilityError(f"unexpected ABI magic 0x{magic:08X}")
    version = AbiVersion.decode(version_word)
    if version.major != ABI_MAJOR:
        raise AbiCompatibilityError(
            f"unsupported ABI major {version.major}; expected {ABI_MAJOR}"
        )
    reported = Capability(_bounded("capabilities", capabilities, 0, 0xFFFFFFFF))
    missing = Capability(required) & ~reported
    if missing:
        raise AbiCompatibilityError(f"missing required capabilities: {missing!s}")
    return version


def prog_len_words(binary_bytes: int) -> int:
    """PROG_LEN for a binary of ``binary_bytes``: 64-bit words, rounded up."""
    return (_bounded("binary_bytes", binary_bytes, 0, 0xFFFFFFFF) + 7) // 8


def in_ddr_window(address: int, size: int) -> bool:
    """Whether [address, address + size) lies in CVA6's DDR window."""
    return DDR_BASE <= address and address + size <= DDR_BASE + DDR_BYTES
