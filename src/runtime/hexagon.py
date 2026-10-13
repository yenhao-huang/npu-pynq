"""Host driver for the Hexagon-style heterogeneous NPU subsystem (issue #136).

The subsystem (``src/hw/rtl/npu_matrix/npu_hexagon/npu_hx_subsystem.sv``) sits
in the hexagon overlay as IP ``npu_hx_0`` (AXI-lite slave at 0x43C10000, 64 KB)
next to the unchanged tensor accelerator ``npu_accelerator_0``. It contains an
RV32I scalar core with L1 I$/D$ and an L2 in front of DDR, a SIMD vector unit
working on a 1024 x 64-bit TCM, and a TCM DMA. The register map below follows
the header of ``npu_hx_regs.sv``, which is authoritative.

Memory model the caller must respect:

- The scalar core fetches and loads/stores DDR through its write-back caches,
  which are not coherent with the ARM or with the TCM DMA. :meth:`HexagonNpu.run`
  issues CONTROL.FLUSH before START (so code or data the host just wrote is
  not shadowed by stale lines) and after the halt (so results reach DDR);
  ``flush_before`` / ``flush_after`` skip one when the program does not need
  it (e.g. a TCM-only kernel whose code is already cached).
- The subsystem's AXI master uses non-coherent HP traffic, so a cacheable
  pynq buffer must be ``flush()``-ed after CPU writes and ``invalidate()``-d
  before CPU reads. :meth:`HexagonNpu.load_program`, :meth:`dma_to_tcm` and
  :meth:`dma_from_tcm` do that for the buffers they are given.
- DDR addresses are physical (the scalar data map sends 0x0000_0000 -
  0x3FFF_FFFF to DDR unchanged), so a program runs from
  ``buffer.physical_address``.

A hung program is recovered with :meth:`HexagonNpu.stop` (CONTROL.STOP,
VERSION >= 0x00010001): the core stops at the next instruction boundary and
reports a halt with cause 2; FAULT_PC is the resume address.
:meth:`HexagonNpu.run` stops the core itself when its timeout expires.

:func:`vector_kernel_source` returns the reference vector kernels (INT8
residual add, FP16 a*x+y) that the board self-test runs; tb_npu_hx_subsystem
simulates the same sources. :class:`VectorOp` keeps one of them resident
for repeated calls with the host overhead cut to the minimum (review
round 3 of the integration; see its docstring for the per-call MMIO count).

Host overhead: every MMIO access from Python costs microseconds, far more
than the subsystem needs for a short kernel. When the MMIO object is a pynq
``DefaultIP`` / ``MMIO`` whose ``array`` maps the whole 64 KB window, the
driver reads and writes that array directly (the same 32-bit loads and
stores pynq's ``MMIO.read`` / ``write`` issue, without their per-call
checks); pass ``fast_mmio=False`` to go through ``read`` / ``write``.

The MMIO object (``read(offset)`` / ``write(offset, value)``) and the buffer
allocator are injectable, so the driver is unit-tested on a host without
pynq (``src/test/tests/test_hexagon_runtime.py``); pynq is imported lazily,
only when no allocator was given and a buffer is needed.
"""

from __future__ import annotations

import functools
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# ------------------------------------------------------------------ registers
# Byte offsets of npu_hx_regs (AXI-lite, 32-bit).
REG_MAGIC = 0x000
REG_VERSION = 0x004
REG_CAPS = 0x008
REG_CONTROL = 0x00C
REG_STATUS = 0x010
REG_START_PC = 0x014
REG_CAUSE = 0x018
REG_FAULT_PC = 0x01C
REG_CYCLES = 0x020
REG_INSTRET = 0x024
REG_L1I_HITS = 0x028
REG_L1I_MISSES = 0x02C
REG_L1D_HITS = 0x030
REG_L1D_MISSES = 0x034
REG_L2_HITS = 0x038
REG_L2_MISSES = 0x03C
REG_DMA_DDR_ADDR = 0x040
REG_DMA_TCM_ADDR = 0x044
REG_DMA_LEN = 0x048
REG_IRQ_ENABLE = 0x04C
REG_MAILBOX0 = 0x050
REG_MAILBOX = (0x050, 0x054, 0x058, 0x05C)
REG_VEC_CYCLES = 0x060
TCM_WINDOW = 0x8000          # 0x8000-0xFFFF: word = offset[14:3], lane = offset[2]
TCM_WINDOW_BYTES = 0x8000
HX_WINDOW_BYTES = 0x10000    # the IP's address range (registers + TCM window)

HX_MAGIC = 0x314E5848        # "HXN1"
HX_VERSION = 0x00010001      # minor 1: CONTROL.STOP, CAUSE[9:8]
HX_VERSION_STOP = 0x00010001  # first VERSION with CONTROL.STOP

# CONTROL bits (write-only pulses)
CONTROL_START = 1 << 0
CONTROL_FLUSH = 1 << 1
CONTROL_DMA_START = 1 << 2
CONTROL_CLEAR = 1 << 3
CONTROL_STOP = 1 << 4        # stop the running scalar core (VERSION >= 0x00010001)

# STATUS bits
STATUS_RUNNING = 1 << 0
STATUS_HALTED = 1 << 1
STATUS_FAULT = 1 << 2
STATUS_VEC_BUSY = 1 << 3
STATUS_DMA_BUSY = 1 << 4
STATUS_FLUSH_BUSY = 1 << 5
STATUS_VEC_ILLEGAL = 1 << 6
STATUS_DMA_ERROR = 1 << 7
STATUS_DMA_DONE = 1 << 8     # sticky
STATUS_BUS_ERROR = 1 << 9    # sticky
STATUS_HALT_EVENT = 1 << 10  # sticky (also set by a STOP), cleared by START (core idle) or CLEAR
STATUS_FAULT_EVENT = 1 << 11  # sticky, cleared by START (core idle) or CLEAR

STATUS_NAMES = {
    STATUS_RUNNING: "running", STATUS_HALTED: "halted", STATUS_FAULT: "fault",
    STATUS_VEC_BUSY: "vec_busy", STATUS_DMA_BUSY: "dma_busy",
    STATUS_FLUSH_BUSY: "flush_busy", STATUS_VEC_ILLEGAL: "vec_illegal",
    STATUS_DMA_ERROR: "dma_error", STATUS_DMA_DONE: "dma_done",
    STATUS_BUS_ERROR: "bus_error", STATUS_HALT_EVENT: "halt_event",
    STATUS_FAULT_EVENT: "fault_event",
}

# DMA_LEN
DMA_LEN_WORDS_MASK = 0xFFFF
DMA_LEN_TO_DDR = 1 << 31

# IRQ_ENABLE
IRQ_HALT = 1 << 0
IRQ_FAULT = 1 << 1
IRQ_DMA_DONE = 1 << 2

# CAPS bits
CAPS_SCALAR = 1 << 0
CAPS_VECTOR = 1 << 1
CAPS_L1I = 1 << 2
CAPS_L1D = 1 << 3
CAPS_L2 = 1 << 4
CAPS_TCM = 1 << 5
CAPS_DMA = 1 << 6

# CAUSE fields and the scalar core's cause codes (rv32_iss FAULT_* / HALT_*)
CAUSE_FAULT_MASK = 0x7
CAUSE_HALT_SHIFT = 8
CAUSE_HALT_MASK = 0x3        # CAUSE[9:8]
HALT_ECALL, HALT_EBREAK, HALT_STOP = 0, 1, 2
FAULT_NAMES = {
    0: "none",
    1: "illegal instruction",
    2: "misaligned instruction fetch / jump target",
    3: "misaligned load",
    4: "misaligned store",
    5: "instruction bus error",
    6: "data bus error (unmapped data address)",
}
HALT_NAMES = {HALT_ECALL: "ecall", HALT_EBREAK: "ebreak", HALT_STOP: "stop"}

CLOCK_HZ = 100_000_000       # FCLK_CLK0 of the overlay
MASK32 = 0xFFFFFFFF


# ------------------------------------------------------------------ errors
class HexagonError(RuntimeError):
    """Any driver-level failure (wrong IP, bad argument, DMA error...)."""


class HexagonTimeout(HexagonError):
    """A program, flush or DMA did not finish within the timeout. When
    :meth:`HexagonNpu.run` stopped the program after its timeout, ``result``
    holds the :class:`RunResult` of the stop (halt cause ``"stop"``)."""

    def __init__(self, message: str, status: int, result: "RunResult | None" = None):
        super().__init__(message)
        self.status = status
        self.result = result


class HexagonFault(HexagonError):
    """The scalar core stopped with a fault; ``result`` holds the details."""

    def __init__(self, result: "RunResult"):
        super().__init__(
            f"scalar core fault: cause {result.fault_cause} ({result.cause_name}) "
            f"at pc=0x{result.fault_pc:08x} after {result.instret} instructions, "
            f"{result.cycles} cycles (STATUS={describe_status(result.status)})")
        self.result = result


# ------------------------------------------------------------------ records
def describe_status(status: int) -> str:
    names = [name for bit, name in STATUS_NAMES.items() if status & bit]
    return f"0x{status:03x}[{','.join(names) or 'idle'}]"


@dataclass(frozen=True)
class Caps:
    raw: int
    scalar: bool
    vector: bool
    l1i: bool
    l1d: bool
    l2: bool
    tcm: bool
    dma: bool
    vlen_bits: int
    tcm_words: int

    @classmethod
    def decode(cls, raw: int) -> "Caps":
        raw &= MASK32
        return cls(raw=raw, scalar=bool(raw & CAPS_SCALAR), vector=bool(raw & CAPS_VECTOR),
                   l1i=bool(raw & CAPS_L1I), l1d=bool(raw & CAPS_L1D), l2=bool(raw & CAPS_L2),
                   tcm=bool(raw & CAPS_TCM), dma=bool(raw & CAPS_DMA),
                   vlen_bits=((raw >> 8) & 0xFF) * 8, tcm_words=((raw >> 16) & 0xFFFF) * 64)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["raw"] = f"0x{self.raw:08x}"
        return d


@dataclass(frozen=True)
class RunResult:
    """Outcome of one scalar program run (from STATUS / CAUSE / counters)."""

    status: int
    halted: bool
    fault: bool
    halt_cause: int
    fault_cause: int
    fault_pc: int          # the stopping PC: ECALL / EBREAK / faulting instruction;
                           # after a STOP the first instruction not executed
    cycles: int
    instret: int
    vec_cycles: int
    seconds: float         # host wall time from START to the observed stop

    @property
    def cause_name(self) -> str:
        if self.fault:
            return FAULT_NAMES.get(self.fault_cause, f"unknown ({self.fault_cause})")
        return HALT_NAMES.get(self.halt_cause, f"unknown ({self.halt_cause})")

    @property
    def stopped(self) -> bool:
        """Halted by CONTROL.STOP; resume with ``start(fault_pc)``."""
        return self.halted and self.halt_cause == HALT_STOP

    @property
    def cpi(self) -> float:
        return self.cycles / self.instret if self.instret else float("inf")

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(status=describe_status(self.status), cause_name=self.cause_name,
                 fault_pc=f"0x{self.fault_pc:08x}", cpi=self.cpi)
        return d


@dataclass(frozen=True)
class Counters:
    """Performance counters. The cache counters saturate and are cleared only
    by reset, so compare two snapshots with :meth:`delta`; CYCLES, INSTRET
    and VEC_CYCLES restart at every START."""

    cycles: int
    instret: int
    l1i_hits: int
    l1i_misses: int
    l1d_hits: int
    l1d_misses: int
    l2_hits: int
    l2_misses: int
    vec_cycles: int

    CACHE_FIELDS = ("l1i_hits", "l1i_misses", "l1d_hits", "l1d_misses", "l2_hits", "l2_misses")

    def delta(self, before: "Counters") -> "Counters":
        """Cache counters relative to ``before``; the run counters as read."""
        diff = {f: (getattr(self, f) - getattr(before, f)) & MASK32 for f in self.CACHE_FIELDS}
        return Counters(cycles=self.cycles, instret=self.instret, vec_cycles=self.vec_cycles, **diff)

    @staticmethod
    def _rate(hits: int, misses: int) -> float | None:
        total = hits + misses
        return hits / total if total else None

    def hit_rates(self) -> dict:
        return {"l1i": self._rate(self.l1i_hits, self.l1i_misses),
                "l1d": self._rate(self.l1d_hits, self.l1d_misses),
                "l2": self._rate(self.l2_hits, self.l2_misses)}

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hit_rates"] = self.hit_rates()
        return d


@dataclass(frozen=True)
class LoadedProgram:
    buffer: Any            # the contiguous buffer holding the program
    phys_addr: int         # physical address of the first instruction (START_PC)
    words: int


class Mailbox:
    """MAILBOX0-3: ``npu.mailbox[i]`` reads, ``npu.mailbox[i] = v`` writes.
    ``mmio`` is anything with ``read(offset)`` / ``write(offset, value)``
    (:class:`HexagonNpu` passes itself)."""

    def __init__(self, mmio: Any):
        self._mmio = mmio

    def __len__(self) -> int:
        return len(REG_MAILBOX)

    def _offset(self, index: int) -> int:
        if not 0 <= index < len(REG_MAILBOX):
            raise IndexError(f"mailbox index {index} out of range 0..{len(REG_MAILBOX) - 1}")
        return REG_MAILBOX[index]

    def __getitem__(self, index: int) -> int:
        return int(self._mmio.read(self._offset(index))) & MASK32

    def __setitem__(self, index: int, value: int) -> None:
        self._mmio.write(self._offset(index), int(value) & MASK32)

    def read_all(self) -> list[int]:
        return [self[i] for i in range(len(self))]


def _bind_mmio(mmio: Any, fast: bool):
    """(read, write) callables for ``mmio``: direct accesses to a pynq
    ``MMIO.array`` (``mmio.array`` or ``mmio.mmio.array``, a uint32 view of
    the whole 64 KB window) when ``fast``, else ``mmio.read`` / ``write``."""
    if fast:
        array = getattr(mmio, "array", None)
        if array is None:
            array = getattr(getattr(mmio, "mmio", None), "array", None)
        if (isinstance(array, np.ndarray) and array.dtype == np.uint32 and array.ndim == 1
                and array.size * 4 >= HX_WINDOW_BYTES):
            get, put = array.__getitem__, array.__setitem__

            def read_fast(offset: int) -> int:
                return int(get(offset >> 2))

            def write_fast(offset: int, value: int) -> None:
                put(offset >> 2, value)

            return read_fast, write_fast

    def read_mmio(offset: int) -> int:
        return int(mmio.read(offset)) & MASK32

    def write_mmio(offset: int, value: int) -> None:
        mmio.write(offset, value)

    return read_mmio, write_mmio


def _default_allocate(shape, dtype):
    from pynq import allocate  # lazy: keeps the module importable on a host

    return allocate(shape=shape, dtype=dtype)


# ------------------------------------------------------------------ driver
class HexagonNpu:
    """Driver for one npu_hx_subsystem instance.

    ``HexagonNpu(overlay)`` binds ``overlay.npu_hx_0``; pass ``mmio=`` to use
    any object with ``read(offset)`` / ``write(offset, value)`` instead
    (``overlay`` may then be None). ``allocate`` creates contiguous buffers
    (default: ``pynq.allocate``); ``clock`` is the monotonic time source used
    for timeouts and wall-time measurements; ``fast_mmio`` allows the direct
    ``MMIO.array`` accesses described in the module docstring.

    Every register write goes through :meth:`write`, which counts it
    (``write_epoch``): a :class:`VectorOp` uses that count to know that
    nothing else touched the subsystem since its previous call.
    """

    def __init__(self, overlay: Any = None, name: str = "npu_hx_0", *, mmio: Any = None,
                 allocate: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.monotonic, check: bool = True,
                 fast_mmio: bool = True):
        if mmio is None:
            if overlay is None:
                raise HexagonError("pass an overlay or an mmio object")
            try:
                mmio = getattr(overlay, name)
            except AttributeError as error:
                raise HexagonError(f"overlay has no IP named {name!r}; "
                                   "is this the --hexagon overlay?") from error
        self.name = name
        self.mmio = mmio
        self._rd, self._wr = _bind_mmio(mmio, fast_mmio)
        self.write_epoch = 0
        self._allocate = allocate or _default_allocate
        self._clock = clock
        self._started: float | None = None
        self.mailbox = Mailbox(self)
        self.magic = self.read(REG_MAGIC)
        self.version = self.read(REG_VERSION)
        self.caps = Caps.decode(self.read(REG_CAPS))
        if check:
            self.check()

    # -------------------------------------------------------------- raw access
    def read(self, offset: int) -> int:
        return self._rd(offset)

    def write(self, offset: int, value: int) -> None:
        self.write_epoch += 1
        self._wr(offset, int(value) & MASK32)

    def status(self) -> int:
        return self._rd(REG_STATUS)

    def check(self) -> None:
        """Raise unless MAGIC / VERSION identify a supported subsystem."""
        if self.magic != HX_MAGIC:
            raise HexagonError(f"{self.name}: MAGIC 0x{self.magic:08x}, expected 0x{HX_MAGIC:08x}")
        if self.version >> 16 != HX_VERSION >> 16:
            raise HexagonError(f"{self.name}: unsupported VERSION 0x{self.version:08x} "
                               f"(driver speaks 0x{HX_VERSION:08x})")
        if not self.caps.scalar:
            raise HexagonError(f"{self.name}: CAPS 0x{self.caps.raw:08x} has no scalar core")

    def info(self) -> dict:
        return {"name": self.name, "magic": f"0x{self.magic:08x}",
                "version": f"0x{self.version:08x}", "caps": self.caps.as_dict()}

    def _poll(self, done: Callable[[int], bool], timeout: float, what: str) -> tuple[int, float]:
        """Read STATUS back to back (no sleep: the subsystem's operations
        take microseconds, about one MMIO read) until ``done(status)``; one
        clock read per extra iteration enforces ``timeout``."""
        read, clock = self._rd, self._clock
        start = clock()
        status = read(REG_STATUS)
        while not done(status):
            if clock() - start > timeout:
                raise HexagonTimeout(f"{self.name}: {what} did not finish within {timeout:g} s "
                                     f"(STATUS={describe_status(status)})", status)
            status = read(REG_STATUS)
        return status, clock() - start

    # -------------------------------------------------------------- control
    def clear(self) -> None:
        """CONTROL.CLEAR: drop dma_done, bus_error, vec_illegal and the
        halt/fault events (acknowledges the interrupt)."""
        self.write(REG_CONTROL, CONTROL_CLEAR)

    def irq_enable(self, mask: int) -> None:
        self.write(REG_IRQ_ENABLE, mask & (IRQ_HALT | IRQ_FAULT | IRQ_DMA_DONE))

    def flush(self, timeout: float = 1.0) -> float:
        """CONTROL.FLUSH: write back L1D then L2 to DDR and invalidate every
        cache (L1I included). Returns the wall time."""
        return self._flush_go(None, timeout)[1]

    def _flush_go(self, status: int | None, timeout: float = 1.0) -> tuple[int, float]:
        """FLUSH; ``status`` is a STATUS value read since the last command
        (skips the busy check read). Returns (final STATUS, wall time)."""
        if (self.status() if status is None else status) & STATUS_FLUSH_BUSY:
            raise HexagonError(f"{self.name}: a flush is already in progress")
        self.write(REG_CONTROL, CONTROL_FLUSH)
        return self._poll(_flush_idle, timeout, "cache flush")

    def start(self, pc: int, *, status: int | None = None) -> None:
        """Start the scalar core at ``pc`` (a DDR physical address).
        ``status``: a STATUS value read since the last command, which
        replaces the read that checks the core is not running."""
        if pc & 3:
            raise HexagonError(f"start pc 0x{pc:08x} is not word aligned")
        if not 0 <= pc < 0x4000_0000:
            raise HexagonError(f"start pc 0x{pc:08x} is outside the DDR window")
        self._start_go(pc, status, True)

    def _start_go(self, pc: int, status: int | None, write_pc: bool) -> None:
        if (self.status() if status is None else status) & STATUS_RUNNING:
            raise HexagonError(f"{self.name}: the scalar core is still running")
        if write_pc:
            self.write(REG_START_PC, pc)
        self._started = self._clock()
        self.write(REG_CONTROL, CONTROL_START)

    def result(self, status: int | None = None, seconds: float = 0.0) -> RunResult:
        """The last program's outcome: STATUS (read unless given), CAUSE,
        FAULT_PC, CYCLES, INSTRET, VEC_CYCLES (5 or 6 reads)."""
        read = self._rd
        status = read(REG_STATUS) if status is None else status
        cause = read(REG_CAUSE)
        return RunResult(
            status=status,
            halted=bool(status & STATUS_HALTED),
            fault=bool(status & STATUS_FAULT),
            halt_cause=(cause >> CAUSE_HALT_SHIFT) & CAUSE_HALT_MASK,
            fault_cause=cause & CAUSE_FAULT_MASK,
            fault_pc=read(REG_FAULT_PC),
            cycles=read(REG_CYCLES),
            instret=read(REG_INSTRET),
            vec_cycles=read(REG_VEC_CYCLES),
            seconds=seconds,
        )

    def wait(self, timeout: float = 10.0, *, raise_on_fault: bool = True) -> RunResult:
        """Wait for the program started by :meth:`start` to halt or fault.

        Completion is taken from the sticky halt/fault events (STATUS bits
        10/11, cleared by START), so a stale ``halted`` from the previous run
        cannot end the wait early. Raises :class:`HexagonTimeout` (the core is
        left running) or, on a fault, :class:`HexagonFault`.
        """
        status, seconds = self._wait_halt(timeout)
        result = self.result(status, seconds)
        if result.fault and raise_on_fault:
            raise HexagonFault(result)
        return result

    def _wait_halt(self, timeout: float) -> tuple[int, float]:
        """Poll for the halt / fault event of the program started last;
        returns (STATUS, wall time since START)."""
        started = self._started
        status, _ = self._poll(_program_done, timeout, "scalar program")
        return status, (self._clock() - started if started is not None else 0.0)

    @property
    def has_stop(self) -> bool:
        """CONTROL.STOP exists (VERSION >= 0x00010001)."""
        return self.version >= HX_VERSION_STOP

    def stop(self, timeout: float = 1.0) -> RunResult:
        """CONTROL.STOP: stop a running (e.g. hung) program at the next
        instruction boundary and return its result. A stopped program reports
        ``halted`` with halt cause 2 (``result.stopped``); ``result.fault_pc``
        is the first instruction not executed, so ``start(result.fault_pc)``
        resumes it (registers are kept; CYCLES / INSTRET restart). A program
        that halts or faults on its own first reports its own cause; on an
        idle core nothing happens and the last result is returned. The vector
        unit and a DMA in flight finish their work. Raises
        :class:`HexagonTimeout` if the core does not stop (a VFENCE or a cache
        access that waits forever: only a bitstream reload helps then)."""
        if not self.has_stop:
            raise HexagonError(f"{self.name}: VERSION 0x{self.version:08x} has no CONTROL.STOP "
                               f"(needs 0x{HX_VERSION_STOP:08x}); reload the bitstream instead")
        self.write(REG_CONTROL, CONTROL_STOP)
        status, seconds = self._poll(lambda s: not s & STATUS_RUNNING, timeout, "STOP")
        return self.result(status, seconds)

    def run(self, pc: int, timeout: float = 10.0, *, flush: bool = True,
            flush_before: bool | None = None, flush_after: bool | None = None,
            raise_on_fault: bool = True, stop_on_timeout: bool = True) -> RunResult:
        """FLUSH, START at ``pc``, wait, FLUSH again so the program's DDR
        stores are visible to the host and the DMA.

        ``flush`` sets both flushes; ``flush_before`` / ``flush_after``
        override one of them. Each FLUSH costs a few MMIO accesses: skip the
        first when the code and the DDR data the program reads are already
        coherent (nothing the caches may hold was rewritten since the last
        FLUSH), the second when the program stores nothing to DDR (e.g. it
        works on the TCM and the mailboxes only).

        When the program does not finish within ``timeout`` and
        ``stop_on_timeout`` is set (and the hardware has CONTROL.STOP), the
        core is stopped and flushed before :class:`HexagonTimeout` is raised;
        the exception's ``result`` holds the stop result. Otherwise the core
        is left running."""
        before = flush if flush_before is None else flush_before
        after = flush if flush_after is None else flush_after
        status = self._flush_go(None)[0] if before else None
        self.start(pc, status=status)
        status, seconds = self._finish(timeout, stop_on_timeout, after)
        result = self.result(status, seconds)
        if result.fault and raise_on_fault:
            raise HexagonFault(result)
        return result

    def _finish(self, timeout: float, stop_on_timeout: bool, flush_after: bool) -> tuple[int, float]:
        """Wait for the program started last, FLUSH after it when asked
        (also after a fault or a STOP); on a timeout STOP it (when enabled
        and supported) and raise :class:`HexagonTimeout`."""
        try:
            status, seconds = self._wait_halt(timeout)
        except HexagonTimeout as error:
            if not (stop_on_timeout and self.has_stop):
                raise
            stopped = self.stop()
            if flush_after:
                self.flush()
            raise HexagonTimeout(f"{error} (stopped at pc=0x{stopped.fault_pc:08x} after "
                                 f"{stopped.instret} instructions)", error.status,
                                 stopped) from error
        if flush_after:
            status = self._flush_go(status)[0]
        return status, seconds

    # -------------------------------------------------------------- buffers
    def allocate(self, words: int, dtype=np.uint32):
        """A physically contiguous DDR buffer of ``words`` elements."""
        return self._allocate(shape=(int(words),), dtype=dtype)

    def load_program(self, words: Sequence[int], buffer: Any = None, offset: int = 0) -> LoadedProgram:
        """Copy 32-bit instruction words into ``buffer`` (allocated when None)
        at word ``offset``, flush the ARM cache for it and return the start
        address. Programs that are not position independent must be assembled
        for ``buffer.physical_address + 4 * offset``."""
        data = np.asarray([int(w) & MASK32 for w in words], dtype=np.uint32)
        if buffer is None:
            buffer = self.allocate(offset + len(data))
        view = _u32_view(buffer)
        if offset < 0 or offset + len(data) > len(view):
            raise HexagonError(f"program of {len(data)} words does not fit at word {offset} "
                               f"of a {len(view)}-word buffer")
        view[offset:offset + len(data)] = data
        _flush(buffer)
        return LoadedProgram(buffer=buffer, phys_addr=int(buffer.physical_address) + 4 * offset,
                             words=len(data))

    # -------------------------------------------------------------- DMA
    def dma(self, ddr_phys: int, tcm_word: int, words: int, to_ddr: bool,
            timeout: float = 1.0) -> float:
        """Copy ``words`` 64-bit words between DDR (physical address) and the
        TCM (word address). Blocks until STATUS.dma_done; raises on
        STATUS.dma_error. Returns the wall time. The caller keeps the ARM
        cache coherent (see :meth:`dma_to_tcm` / :meth:`dma_from_tcm`)."""
        if not self.caps.dma:
            raise HexagonError(f"{self.name}: CAPS reports no TCM DMA")
        if ddr_phys & 7:
            raise HexagonError(f"DMA DDR address 0x{ddr_phys:08x} is not 8-byte aligned")
        if not 1 <= words <= DMA_LEN_WORDS_MASK:
            raise HexagonError(f"DMA length {words} words outside 1..{DMA_LEN_WORDS_MASK}")
        if tcm_word < 0 or tcm_word + words > self.caps.tcm_words:
            raise HexagonError(f"DMA TCM range [{tcm_word}, {tcm_word + words}) outside the "
                               f"{self.caps.tcm_words}-word TCM")
        return self._dma_go(ddr_phys, tcm_word, words, to_ddr, None, timeout)[1]

    def _dma_go(self, ddr_phys: int, tcm_word: int, words: int, to_ddr: bool,
                status: int | None, timeout: float = 1.0) -> tuple[int, float]:
        """The DMA of :meth:`dma` with checked arguments; ``status`` is a
        STATUS value read since the last command (skips the busy check
        read). Returns (final STATUS, wall time)."""
        if (self.status() if status is None else status) & STATUS_DMA_BUSY:
            raise HexagonError(f"{self.name}: the DMA is busy")
        self.write(REG_DMA_DDR_ADDR, ddr_phys)
        self.write(REG_DMA_TCM_ADDR, tcm_word)
        self.write(REG_DMA_LEN, (DMA_LEN_TO_DDR if to_ddr else 0) | words)
        self.write(REG_CONTROL, CONTROL_DMA_START)   # also clears the sticky dma_done
        status, seconds = self._poll(_dma_finished, timeout, "TCM DMA")
        if status & STATUS_DMA_ERROR:
            raise HexagonError(f"{self.name}: DMA error ({'TCM->DDR' if to_ddr else 'DDR->TCM'}, "
                               f"ddr=0x{ddr_phys:08x} tcm={tcm_word} words={words}, "
                               f"STATUS={describe_status(status)})")
        return status, seconds

    def dma_to_tcm(self, buffer: Any, tcm_word: int, words: int | None = None,
                   offset_bytes: int = 0, timeout: float = 1.0) -> float:
        """Flush ``buffer`` and DMA it (from ``offset_bytes``) into the TCM."""
        words = _buffer_words64(buffer, offset_bytes) if words is None else words
        _flush(buffer)
        return self.dma(int(buffer.physical_address) + offset_bytes, tcm_word, words, False, timeout)

    def dma_from_tcm(self, buffer: Any, tcm_word: int, words: int | None = None,
                     offset_bytes: int = 0, timeout: float = 1.0) -> float:
        """DMA TCM words into ``buffer`` (at ``offset_bytes``) and invalidate it."""
        words = _buffer_words64(buffer, offset_bytes) if words is None else words
        seconds = self.dma(int(buffer.physical_address) + offset_bytes, tcm_word, words, True, timeout)
        _invalidate(buffer)
        return seconds

    # -------------------------------------------------------------- TCM window
    def _check_tcm(self, word: int, count: int) -> None:
        limit = min(self.caps.tcm_words, TCM_WINDOW_BYTES // 8)
        if word < 0 or count < 0 or word + count > limit:
            raise HexagonError(f"TCM range [{word}, {word + count}) outside the {limit}-word window")

    def tcm_read(self, word: int, count: int = 1) -> list[int]:
        """Read ``count`` 64-bit TCM words through the host window (slow:
        two AXI-lite reads per word; use the DMA for bulk data)."""
        self._check_tcm(word, count)
        read = self._rd
        out = []
        for w in range(word, word + count):
            lo = read(TCM_WINDOW + 8 * w)
            hi = read(TCM_WINDOW + 8 * w + 4)
            out.append(lo | (hi << 32))
        return out

    def tcm_write(self, word: int, values: Iterable[int]) -> None:
        """Write 64-bit words into the TCM through the host window."""
        values = [int(v) & ((1 << 64) - 1) for v in values]
        self._check_tcm(word, len(values))
        for i, v in enumerate(values):
            self.write(TCM_WINDOW + 8 * (word + i), v & MASK32)
            self.write(TCM_WINDOW + 8 * (word + i) + 4, v >> 32)

    # -------------------------------------------------------------- counters
    def counters(self) -> Counters:
        r = self._rd
        return Counters(cycles=r(REG_CYCLES), instret=r(REG_INSTRET),
                        l1i_hits=r(REG_L1I_HITS), l1i_misses=r(REG_L1I_MISSES),
                        l1d_hits=r(REG_L1D_HITS), l1d_misses=r(REG_L1D_MISSES),
                        l2_hits=r(REG_L2_HITS), l2_misses=r(REG_L2_MISSES),
                        vec_cycles=r(REG_VEC_CYCLES))


# ------------------------------------------------------------------ kernels
VECTOR_KERNELS = ("int8", "fp16")
_KX = ("a0", "a1", "a2", "a3")        # x pointers, one per unrolled word
_KY = ("a4", "a5", "a6", "a7")        # r / y pointers
_KO = ("t3", "t4", "t5", "t6")        # result pointers


def vector_kernel_source(kind: str, unroll: int = 4) -> str:
    """RV32I + custom-0 assembly (src/test/model/rv32_asm.py) of a reference
    vector kernel over the TCM, position independent.

    Arguments come from the mailboxes: MAILBOX0 = W, the 64-bit TCM words per
    operand (a multiple of ``unroll``, at least ``unroll``); MAILBOX1 = the
    FP16 coefficient ``a`` (binary16 bits). TCM layout: x at word 0, r / y at
    word W, the result at word 2W.

    - ``"int8"``: y = saturate_int8(x + r), 8 lanes per word.
    - ``"fp16"``: y = a * x + y in binary16 (vmul then vadd, two roundings).

    The scalar core sends at most one instruction every 3 cycles and the
    vector unit never back-pressures these kernels, so their time is the
    scalar instruction count: per word 4 (int8) / 5 (fp16) vector commands
    plus 3 pointer bumps (every command needs its address in a register),
    plus 1 / ``unroll`` for the loop branch. ``unroll`` words are processed
    per iteration in software-pipelined order (all loads, then the
    arithmetic, then the stores), so the vector unit's 5-cycle dependency and
    VST drain latencies stay hidden."""
    if kind not in VECTOR_KERNELS:
        raise ValueError(f"kind must be one of {VECTOR_KERNELS}")
    if unroll not in (1, 2, 4):
        raise ValueError("unroll must be 1, 2 or 4")
    u = unroll
    lines = ["    lui   s0, 0x80000         # npu_hx_regs through the scalar MMIO window",
             "    lw    t2, 0x50(s0)        # MAILBOX0 = W (loop end for the x pointer)",
             "    slli  t0, t2, 1           # 2W"]
    if kind == "fp16":
        lines += ["    lw    t1, 0x54(s0)        # MAILBOX1 = a",
                  "    vsplat.f16 v0, t1"]
    for k in range(u):
        lines += [f"    addi  {_KX[k]}, x0, {k}",
                  f"    addi  {_KY[k]}, t2, {k}",
                  f"    addi  {_KO[k]}, t0, {k}"]
    lines.append("loop:")
    body = []
    if kind == "int8":
        # x_k -> v(2k), r_k -> v(2k+1), result in v(2k)
        for k in range(u):
            body += [f"    vld   v{2 * k}, {_KX[k]}", f"    vld   v{2 * k + 1}, {_KY[k]}"]
        for k in range(u):
            body += [f"    vadd.i8 v{2 * k}, v{2 * k}, v{2 * k + 1}", f"    addi  {_KX[k]}, {_KX[k]}, {u}"]
        for k in range(u):
            body += [f"    vst   v{2 * k}, {_KO[k]}", f"    addi  {_KY[k]}, {_KY[k]}, {u}"]
    else:
        # x_k -> v(1+k) (a * x in place), y_k -> v(5 + k % 3), result in v(1+k)
        for k in range(u):
            body.append(f"    vld   v{1 + k}, {_KX[k]}")
        for k in range(u):
            body += [f"    vmul.f16 v{1 + k}, v0, v{1 + k}", f"    addi  {_KX[k]}, {_KX[k]}, {u}"]
        for k in range(u):
            body.append(f"    vld   v{5 + k % 3}, {_KY[k]}")
            if k:
                body.append(f"    vadd.f16 v{k}, v{k}, v{5 + (k - 1) % 3}")
        body.append(f"    vadd.f16 v{u}, v{u}, v{5 + (u - 1) % 3}")
        for k in range(u):
            body += [f"    vst   v{1 + k}, {_KO[k]}", f"    addi  {_KY[k]}, {_KY[k]}, {u}"]
    body += [f"    addi  {_KO[k]}, {_KO[k]}, {u}" for k in range(u)]
    body.append(f"    bne   {_KX[0]}, t2, loop")
    lines += body + ["    vfence", "    ecall", ""]
    return "\n".join(lines)


@functools.lru_cache(maxsize=None)
def vector_kernel_words(kind: str, unroll: int = 4) -> tuple[int, ...]:
    """The instruction words of :func:`vector_kernel_source`, assembled once
    per (kind, unroll) and cached (the assembler, src/test/model/rv32_asm.py,
    is imported on first use). Position independent: load them anywhere."""
    from src.test.model.rv32_asm import assemble  # lazy: only needed to build kernels

    return tuple(assemble(vector_kernel_source(kind, unroll), base=0).words)


class VectorOp:
    """A reference vector kernel (:func:`vector_kernel_source`) kept resident
    with its staging buffers, for repeated calls at minimum host overhead.

    ``VectorOp(npu, "int8", max_elements)`` loads the kernel into its own
    DDR buffer once and allocates an input buffer (x | y) and an output
    buffer for up to ``max_elements`` lanes. ``op(x, y)`` (FP16:
    ``op(x, y, a)``) copies the operands in, DMAs them into the TCM (x at
    word 0, y at W, result at 2W; W = 64-bit words per operand), runs the
    kernel, DMAs the result back and returns it as a new array (or into
    ``out``). Bit-exact with :func:`vector_kernel_source` run through
    :meth:`HexagonNpu.run`.

    What a call skips compared with ``dma_to_tcm`` + ``run`` +
    ``dma_from_tcm``: the kernel touches only the TCM and the mailboxes, so
    no FLUSH after it; its code stays valid in the caches, so no FLUSH
    before it either, except on the first call (the program was just
    written) and whenever another register write reached the subsystem
    since this op's previous call (``HexagonNpu.write_epoch`` changed). In
    that undisturbed steady state the core is known to be halted, the DMA
    idle and START_PC / MAILBOX0-1 to hold this op's values, so the STATUS
    check reads and the unchanged writes are skipped too, every STATUS
    value a poll returns is reused as the next step's check, and only
    CYCLES is read of the run's result (``stats=True`` reads all of
    :meth:`HexagonNpu.result`). MMIO accesses per steady call: 9 writes
    (DMA_DDR_ADDR, DMA_TCM_ADDR, DMA_LEN, CONTROL twice for the DMAs,
    CONTROL.START) and 4 reads (three completion polls that each end on
    their first STATUS read when the hardware is faster than one MMIO
    access, and CYCLES), against 14 writes and 15 reads for the generic
    sequence; plus one buffer flush and one invalidate (the pynq cache
    maintenance the non-coherent DMA needs). Timeouts are those of the
    generic calls: a program that does not halt within ``timeout`` is
    stopped (CONTROL.STOP) and :class:`HexagonTimeout` raised; the next
    call then takes the full path.

    The op owns MAILBOX0-1, START_PC and TCM words 0..3W-1 during a call.
    Do not write its program or staging buffers yourself.
    """

    _LANES = {"int8": (np.int8, 8), "fp16": (np.float16, 4)}

    def __init__(self, npu: HexagonNpu, kind: str, max_elements: int, *, unroll: int = 4,
                 words: Sequence[int] | None = None, timeout: float = 10.0,
                 dma_timeout: float = 1.0):
        if kind not in VECTOR_KERNELS:
            raise ValueError(f"kind must be one of {VECTOR_KERNELS}")
        if not npu.caps.vector or not npu.caps.dma:
            raise HexagonError(f"{npu.name}: VectorOp needs the vector unit and the TCM DMA "
                               f"(CAPS 0x{npu.caps.raw:08x})")
        self.npu, self.kind, self.unroll = npu, kind, unroll
        self.dtype, self.lanes = self._LANES[kind]
        self.max_words = self._words_of(max_elements)
        if 3 * self.max_words > npu.caps.tcm_words:
            raise HexagonError(f"{max_elements} {kind} elements need {3 * self.max_words} TCM "
                               f"words (> {npu.caps.tcm_words})")
        self.timeout, self.dma_timeout = timeout, dma_timeout
        program = vector_kernel_words(kind, unroll) if words is None else tuple(words)
        self.program = npu.load_program(program)
        self._in = npu.allocate(16 * self.max_words, dtype=np.uint8)
        self._out = npu.allocate(8 * self.max_words, dtype=np.uint8)
        self._in_bytes = np.asarray(self._in).view(np.uint8)
        self._out_bytes = np.asarray(self._out).view(np.uint8)
        self._in_phys = int(self._in.physical_address)
        self._out_phys = int(self._out.physical_address)
        self._epoch: int | None = None   # write_epoch after this op's last call
        self._status = 0                 # STATUS at the end of that call
        self._args: tuple[int, int] | None = None
        self.calls = 0
        self.last_cycles: int | None = None
        self.last_result: RunResult | None = None
        self.last_profile: dict | None = None

    def _words_of(self, elements: int) -> int:
        per_iter = self.lanes * self.unroll
        if elements <= 0 or elements % per_iter:
            raise HexagonError(f"{self.kind} element count {elements} must be a positive "
                               f"multiple of {per_iter} ({self.lanes} lanes x unroll {self.unroll})")
        return elements // self.lanes

    def free(self) -> None:
        """Release the program and staging buffers (pynq ``freebuffer``)."""
        for buf in (self.program.buffer, self._in, self._out):
            free = getattr(buf, "freebuffer", None)
            if free is not None:
                free()

    def __call__(self, x: np.ndarray, y: np.ndarray, a: float = 0.0, *,
                 out: np.ndarray | None = None, stats: bool = False,
                 profile: bool = False) -> np.ndarray:
        """INT8: saturate(x + y); FP16: a * x + y (``a`` rounded to binary16).
        ``stats`` stores the full :class:`RunResult` in ``last_result``;
        ``profile`` stores the wall time of each phase in ``last_profile``."""
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        if x.dtype != self.dtype or y.dtype != self.dtype or x.shape != y.shape:
            raise HexagonError(f"{self.kind} VectorOp needs two {np.dtype(self.dtype).name} "
                               f"arrays of one shape (got {x.dtype}{x.shape}, {y.dtype}{y.shape})")
        w = self._words_of(x.size)
        if w > self.max_words:
            raise HexagonError(f"{x.size} elements exceed this op's {self.max_words * self.lanes}")
        if out is not None and (out.dtype != self.dtype or out.size != x.size
                                or not out.flags.c_contiguous):
            raise HexagonError(f"out must be a contiguous {np.dtype(self.dtype).name} array of "
                               f"{x.size} elements")
        a_bits = int(np.array([a], dtype=np.float16).view(np.uint16)[0]) if self.kind == "fp16" else 0
        npu = self.npu
        tick = time.perf_counter if profile else None
        t0 = tick() if tick else 0.0
        fast = self._epoch is not None and self._epoch == npu.write_epoch
        self._epoch = None                       # until this call completes
        n8 = 8 * w
        self._in_bytes[:n8] = x.reshape(-1).view(np.uint8)
        self._in_bytes[n8:2 * n8] = y.reshape(-1).view(np.uint8)
        _flush(self._in)
        if fast:
            status = self._status
        else:
            status = npu.status()
            if status & (STATUS_RUNNING | STATUS_DMA_BUSY | STATUS_FLUSH_BUSY):
                raise HexagonError(f"{npu.name}: subsystem busy (STATUS={describe_status(status)})")
            self._args = None
        sticky = status & (STATUS_VEC_ILLEGAL | STATUS_BUS_ERROR)
        if self._args != (w, a_bits):
            npu.write(REG_MAILBOX[0], w)
            npu.write(REG_MAILBOX[1], a_bits)
            self._args = (w, a_bits)
        t1 = tick() if tick else 0.0
        status, _ = npu._dma_go(self._in_phys, 0, 2 * w, False, status, self.dma_timeout)
        t2 = tick() if tick else 0.0
        if not fast:
            status, _ = npu._flush_go(status)    # the program was just written / others ran
        npu._start_go(self.program.phys_addr, status, not fast)
        status, seconds = npu._finish(self.timeout, True, False)
        if stats or status & STATUS_FAULT:
            result = npu.result(status, seconds)
            self.last_result = result
            self.last_cycles = result.cycles
            if result.fault:
                raise HexagonFault(result)
        else:
            self.last_cycles = npu.read(REG_CYCLES)
        new = status & (STATUS_VEC_ILLEGAL | STATUS_BUS_ERROR) & ~sticky
        if new:
            raise HexagonError(f"{npu.name}: {self.kind} kernel raised "
                               f"STATUS={describe_status(status)}")
        t3 = tick() if tick else 0.0
        status, _ = npu._dma_go(self._out_phys, 2 * w, w, True, status, self.dma_timeout)
        _invalidate(self._out)
        result_bytes = self._out_bytes[:n8]
        if out is None:
            out = result_bytes.view(self.dtype).reshape(x.shape).copy()
        else:
            out.reshape(-1).view(np.uint8)[:] = result_bytes
        self._status = status
        self._epoch = npu.write_epoch
        self.calls += 1
        if tick:
            t4 = tick()
            self.last_profile = {"fast_path": fast, "stage_and_flush": t1 - t0,
                                 "dma_in": t2 - t1, "run": t3 - t2,
                                 "dma_out_and_invalidate": t4 - t3, "total": t4 - t0}
        return out


# ------------------------------------------------------------------ helpers
def _flush_idle(status: int) -> bool:
    return not status & STATUS_FLUSH_BUSY


def _program_done(status: int) -> bool:
    # the sticky events (cleared by START) and not running: a stale
    # ``halted`` of the previous run cannot end the wait early
    return bool(status & (STATUS_HALT_EVENT | STATUS_FAULT_EVENT)) and not status & STATUS_RUNNING


def _dma_finished(status: int) -> bool:
    return bool(status & STATUS_DMA_DONE) and not status & STATUS_DMA_BUSY


def _u32_view(buffer: Any) -> np.ndarray:
    array = buffer if isinstance(buffer, np.ndarray) else np.asarray(buffer)
    if array.dtype != np.uint32:
        array = array.view(np.uint32)
    return array


def _buffer_words64(buffer: Any, offset_bytes: int) -> int:
    nbytes = int(getattr(buffer, "nbytes", np.asarray(buffer).nbytes)) - offset_bytes
    if nbytes <= 0 or nbytes % 8:
        raise HexagonError(f"buffer tail of {nbytes} bytes is not a whole number of 64-bit words")
    return nbytes // 8


def _flush(buffer: Any) -> None:
    flush = getattr(buffer, "flush", None)
    if flush is not None:
        flush()


def _invalidate(buffer: Any) -> None:
    invalidate = getattr(buffer, "invalidate", None)
    if invalidate is not None:
        invalidate()


__all__ = [name for name in dir() if name.isupper() and not name.startswith("_")] + [
    "Caps", "Counters", "HexagonError", "HexagonFault", "HexagonNpu", "HexagonTimeout",
    "LoadedProgram", "Mailbox", "RunResult", "VectorOp", "describe_status",
    "vector_kernel_source", "vector_kernel_words",
]
