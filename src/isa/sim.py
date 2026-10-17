"""Bit-accurate instruction-level simulator for the NPU ISA.

``run`` executes a program against a byte array standing in for the data
region at DATA_BASE. It is the golden model the RTL testbench, the board
runtime and the host runtime are compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import isa


@dataclass
class Trace:
    instructions: int = 0
    jobs: int = 0
    macs: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    jobs_log: list[tuple[int, int, int, int, int, int]] = field(default_factory=list)


class ISAFault(RuntimeError):
    def __init__(self, code: int, pc: int, message: str) -> None:
        self.code = code
        self.pc = pc
        super().__init__(f"pc={pc}: {message} (error 0x{code:02x})")


def run(
    program: list[int] | bytes,
    memory: np.ndarray,
    limits: isa.Limits = isa.Limits(),
    *,
    max_instructions: int = 1 << 26,
    log_jobs: bool = False,
) -> Trace:
    """Execute ``program`` on ``memory`` (a writable uint8 array) in place."""
    words = isa.unpack(program) if isinstance(program, (bytes, bytearray)) else list(program)
    if memory.dtype != np.uint8 or memory.ndim != 1:
        raise TypeError("memory must be a one-dimensional uint8 array")
    mem_i8 = memory.view(np.int8)
    m = n = k = 0
    a = b = c = 0
    a_inc = b_inc = c_inc = 0
    trace = Trace()
    pc = 0
    repeat_left = 0
    while True:
        if pc >= len(words):
            raise ISAFault(isa.ERR_FETCH, pc, "program ran past its end without END")
        if trace.instructions >= max_instructions:
            raise ISAFault(isa.ERR_FETCH, pc, "instruction budget exhausted")
        inst = isa.decode(words[pc])
        f = inst.fields()
        count = 1
        if inst.opcode == isa.OP_REPEAT:
            if pc + 1 >= len(words):
                raise ISAFault(isa.ERR_FETCH, pc, "REPEAT is the last word")
            count = f["count"]
            pc += 1
            trace.instructions += 1
            inst = isa.decode(words[pc])
            f = inst.fields()
        for _ in range(count):
            trace.instructions += 1
            op = inst.opcode
            if op in (isa.OP_NOP, isa.OP_FENCE):
                pass
            elif op == isa.OP_END:
                return trace
            elif op == isa.OP_SHAPE:
                m, n, k = f["m"], f["n"], f["k"]
            elif op == isa.OP_ADDR_A:
                a = f["offset"]
            elif op == isa.OP_ADDR_B:
                b = f["offset"]
            elif op == isa.OP_ADDR_C:
                c = f["offset"]
            elif op == isa.OP_INCR:
                a_inc, b_inc, c_inc = f["a_inc"], f["b_inc"], f["c_inc"]
            elif op == isa.OP_GEMM:
                if not (1 <= m <= limits.rows and 1 <= n <= limits.columns and 1 <= k <= limits.max_k):
                    raise ISAFault(isa.ERR_BAD_SHAPE, pc, f"shape m={m} n={n} k={k} exceeds limits")
                if (a | b | c) % isa.ALIGN:
                    raise ISAFault(isa.ERR_MISALIGNED, pc, "tile offset is not 8-byte aligned")
                a_end, b_end, c_end = a + m * k, b + k * n, c + 4 * m * n
                if max(a_end, b_end, c_end) > memory.size:
                    raise ISAFault(isa.ERR_AXI_READ, pc, "tile lies outside the data region")
                lhs = mem_i8[a:a_end].reshape(m, k).astype(np.int32)
                rhs = mem_i8[b:b_end].reshape(k, n).astype(np.int32)
                memory[c:c_end] = (lhs @ rhs).astype("<i4").view(np.uint8).reshape(-1)
                trace.jobs += 1
                trace.macs += m * n * k
                trace.bytes_read += m * k + k * n
                trace.bytes_written += 4 * m * n
                if log_jobs:
                    trace.jobs_log.append((m, n, k, a, b, c))
                flags = f["flags"]
                if flags & isa.GEMM_INC_A:
                    a = (a + a_inc) & 0xFFFFFFFF
                if flags & isa.GEMM_INC_B:
                    b = (b + b_inc) & 0xFFFFFFFF
                if flags & isa.GEMM_INC_C:
                    c = (c + c_inc) & 0xFFFFFFFF
            else:
                raise ISAFault(isa.ERR_ILLEGAL_OPCODE, pc, f"illegal opcode 0x{op:02x}")
        pc += 1
