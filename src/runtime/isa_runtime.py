"""Run NPU instruction-stream programs (src/isa) on the PYNQ overlay.

The program and every tile it touches live in physically contiguous buffers.
``IsaDevice.run`` points PROG_ADDR/PROG_LEN/DATA_BASE at them, starts the
front end, and polls ISA_STATUS until the last job has been written back.
Buffers are cacheable, so callers flush after CPU writes and invalidate before
CPU reads; ``gemm`` shows the full sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable
import numpy as np
from src.isa import isa, layout


REG_VERSION = 0x04
REG_CAPABILITIES = 0x08
REG_ISA_CONTROL = 0x40
REG_ISA_STATUS = 0x44
REG_PROG_ADDR = 0x48
REG_PROG_LEN = 0x4C
REG_DATA_BASE = 0x50
REG_ISA_ERROR = 0x54
REG_ISA_PC = 0x58
REG_ISA_JOBS = 0x5C
REG_ISA_CYCLES = 0x60
REG_ISA_INSTRUCTIONS = 0x64

CAP_ISA_FRONTEND = 1 << 6
STATUS_RUNNING = 1
STATUS_DONE = 2
STATUS_ERROR = 4
STATUS_BUSY = 8


class IsaRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunStats:
    cycles: int
    jobs: int
    instructions: int
    seconds: float


class IsaDevice:
    def __init__(self, overlay: Any, accelerator_name: str = "npu_accelerator_0") -> None:
        self.mmio = getattr(overlay, accelerator_name)
        if not int(self.mmio.read(REG_CAPABILITIES)) & CAP_ISA_FRONTEND:
            raise IsaRuntimeError("overlay does not implement the ISA front end (ABI 1.1)")
        params = overlay.ip_dict[accelerator_name].get("parameters", {})
        self.limits = isa.Limits(
            rows=int(params.get("ROWS", 16)),
            columns=int(params.get("COLUMNS", 16)),
            max_k=int(params.get("MAX_K", 256)),
        )

    def run(self, prog_addr: int, prog_len: int, data_base: int, *, timeout: float = 30.0,
            monotonic: Callable[[], float] = time.monotonic) -> RunStats:
        if prog_addr % 8 or data_base % 8:
            raise IsaRuntimeError("program and data must be 8-byte aligned")
        if int(self.mmio.read(REG_ISA_STATUS)) & (STATUS_RUNNING | STATUS_BUSY):
            raise IsaRuntimeError("the front end is still busy")
        self.mmio.write(REG_PROG_ADDR, prog_addr)
        self.mmio.write(REG_PROG_LEN, prog_len)
        self.mmio.write(REG_DATA_BASE, data_base)
        start = monotonic()
        self.mmio.write(REG_ISA_CONTROL, 1)
        while True:
            status = int(self.mmio.read(REG_ISA_STATUS))
            if not status & (STATUS_RUNNING | STATUS_BUSY):
                break
            if monotonic() - start > timeout:
                raise IsaRuntimeError(f"program timed out at pc={int(self.mmio.read(REG_ISA_PC))}")
        elapsed = monotonic() - start
        if status & STATUS_ERROR or not status & STATUS_DONE:
            code = int(self.mmio.read(REG_ISA_ERROR))
            pc = int(self.mmio.read(REG_ISA_PC))
            raise IsaRuntimeError(f"program failed: ISA_ERROR=0x{code:02x} pc={pc}")
        return RunStats(
            cycles=int(self.mmio.read(REG_ISA_CYCLES)),
            jobs=int(self.mmio.read(REG_ISA_JOBS)),
            instructions=int(self.mmio.read(REG_ISA_INSTRUCTIONS)),
            seconds=elapsed,
        )


def gemm(device: IsaDevice, allocate: Callable[..., Any], a: np.ndarray, b: np.ndarray,
         tk: int | None = None) -> tuple[np.ndarray, RunStats]:
    """``a @ b`` (INT8 x INT8 -> INT32) through one instruction-stream program."""
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dimensions differ")
    plan = layout.plan_gemm(m, n, k, device.limits, tk)
    a_base = 0
    b_base = isa.align(plan.a_bytes, 64)
    c_base = b_base + isa.align(plan.b_bytes, 64)
    data = allocate(shape=(c_base + plan.c_bytes,), dtype=np.uint8)
    data[a_base:a_base + plan.a_bytes] = layout.pack_a(plan, a).view(np.uint8)
    data[b_base:b_base + plan.b_bytes] = layout.pack_b(plan, b).view(np.uint8)
    words = layout.encode(plan, a_base, b_base, c_base) + [isa.end()]
    program = allocate(shape=(len(words),), dtype=np.uint64)
    program[:] = np.asarray(words, dtype=np.uint64)
    data.flush()
    program.flush()
    try:
        stats = device.run(program.physical_address, len(words), data.physical_address)
        data.invalidate()
        partial = layout.unpack_c(plan, np.asarray(data[c_base:]))
        return partial.sum(axis=0, dtype=np.int64).astype(np.int32), stats
    finally:
        data.freebuffer()
        program.freebuffer()


class NpuGemm:
    def __init__(self, overlay):
        from pynq import allocate
        self.allocate = allocate
        self.device = IsaDevice(overlay)
        self.data = allocate(shape=(4 << 20,), dtype=np.uint8)
        self.program = allocate(shape=(4096,), dtype=np.uint64)
        self.stats = {"calls": 0, "jobs": 0, "cycles": 0, "instructions": 0}

    def __call__(self, a, b):
        m, k = a.shape
        plan = layout.plan_gemm(m, b.shape[1], k, self.device.limits)
        b_base = isa.align(plan.a_bytes, 64)
        # Partial outputs are retained for each K chunk, then reduced exactly.
        stage = (len(self.data) - b_base - plan.c_bytes - 64)
        if stage < plan.b_panel:
            if m > self.device.limits.rows:
                # Quantization was already computed over the complete activation.
                # Splitting integer rows preserves that scale and zero point exactly.
                return np.concatenate([self(a[start:start+self.device.limits.rows],b)
                                       for start in range(0,m,self.device.limits.rows)],axis=0)
            raise MemoryError("GEMM activation/output exceeds the bounded CMA window")
        packed = layout.pack_b(plan, b)
        ab = layout.pack_a(plan, a).view(np.uint8)
        c_base = isa.align(b_base + min(stage, plan.b_bytes), 64)
        self.data[:len(ab)] = ab
        partial = np.zeros((plan.kt, plan.m, plan.n), np.int32)
        for lo, hi in plan.segments(c_base - b_base):
            size = (hi - lo) * plan.b_panel
            self.data[b_base:b_base+size] = packed[lo*plan.b_panel:hi*plan.b_panel].view(np.uint8)
            words = layout.encode(plan, 0, b_base, c_base, lo, hi) + [isa.end()]
            if len(words) > len(self.program):
                raise MemoryError("GEMM program exceeds instruction window")
            self.program[:len(words)] = words
            self.data[:c_base+plan.c_bytes].flush()
            self.program[:len(words)].flush()
            stats = self.device.run(self.program.physical_address, len(words), self.data.physical_address)
            self.data[c_base:c_base+plan.c_bytes].invalidate()
            values = layout.unpack_c(plan, np.asarray(self.data[c_base:c_base+plan.c_bytes]))
            partial[:, :, lo*plan.tn:min(hi*plan.tn, plan.n)] = values[:, :, lo*plan.tn:min(hi*plan.tn, plan.n)]
            for key in ("jobs", "cycles", "instructions"):
                self.stats[key] += getattr(stats, key)
        self.stats["calls"] += 1
        return partial.sum(axis=0, dtype=np.int64).astype(np.int32)

    def close(self):
        self.data.freebuffer()
        self.program.freebuffer()
