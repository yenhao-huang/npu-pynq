"""NPU instruction set, version 1.

Every instruction is one little-endian 64-bit word. Bits [63:56] hold the
opcode; the low bits hold its operands. The front end fetches words from
``PROG_ADDR + 8 * pc`` (IF), decodes them against the architectural state
below (ID), and issues matrix jobs to the controller and load/store unit (EX).

Architectural state, all reset by a program start:

``m, n, k``          job shape set by SHAPE
``a, b, c``          byte offsets from DATA_BASE of the A, B and C tiles
``a_inc, b_inc, c_inc``  post-increment amounts, in bytes, set by INCR

A GEMM reads the dense row-major ``m x k`` signed INT8 tile at ``a`` and the
``k x n`` tile at ``b``, and writes the ``m x n`` INT32 result row-major at
``c``. Offsets are 8-byte aligned. ``flags`` selects which of ``a``, ``b``,
``c`` advance by their increment after the job is issued. REPEAT executes the
next instruction ``count`` times, which with post-increment walks a row of
tiles in two words. END waits until every job has been written back.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

ISA_VERSION = 1

OP_NOP = 0x00
OP_END = 0x01
OP_FENCE = 0x02
OP_SHAPE = 0x10
OP_ADDR_A = 0x11
OP_ADDR_B = 0x12
OP_ADDR_C = 0x13
OP_INCR = 0x14
OP_REPEAT = 0x15
OP_GEMM = 0x20

OPCODES = {
    "nop": OP_NOP,
    "end": OP_END,
    "fence": OP_FENCE,
    "shape": OP_SHAPE,
    "addr_a": OP_ADDR_A,
    "addr_b": OP_ADDR_B,
    "addr_c": OP_ADDR_C,
    "incr": OP_INCR,
    "repeat": OP_REPEAT,
    "gemm": OP_GEMM,
}
NAMES = {value: name for name, value in OPCODES.items()}

GEMM_INC_A = 1
GEMM_INC_B = 2
GEMM_INC_C = 4

# Front-end error codes, reported in ISA_ERROR. They do not overlap the
# matrix controller's codes (1-6), which ISA_ERROR also forwards.
ERR_NONE = 0x00
ERR_ILLEGAL_OPCODE = 0x10
ERR_BAD_SHAPE = 0x11
ERR_MISALIGNED = 0x12
ERR_AXI_READ = 0x13
ERR_AXI_WRITE = 0x14
ERR_FETCH = 0x15

ALIGN = 8
INCR_UNIT = 8  # INCR fields count 8-byte units


class ISAError(ValueError):
    """An instruction or program violates the ISA contract."""


@dataclass(frozen=True)
class Limits:
    """Physical limits of one implementation, read from the overlay."""

    rows: int = 16
    columns: int = 16
    max_k: int = 256


def _field(value: int, bits: int, name: str) -> int:
    if not 0 <= value < (1 << bits):
        raise ISAError(f"{name}={value} does not fit in {bits} bits")
    return value


def _word(opcode: int, payload: int) -> int:
    return (opcode << 56) | payload


def nop() -> int:
    return _word(OP_NOP, 0)


def end() -> int:
    return _word(OP_END, 0)


def fence() -> int:
    return _word(OP_FENCE, 0)


def shape(m: int, n: int, k: int) -> int:
    return _word(OP_SHAPE, (_field(m, 8, "m") << 24) | (_field(n, 8, "n") << 16) | _field(k, 16, "k"))


def _offset(value: int, name: str) -> int:
    _field(value, 32, name)
    if value % ALIGN:
        raise ISAError(f"{name}=0x{value:x} is not {ALIGN}-byte aligned")
    return value


def addr_a(offset: int) -> int:
    return _word(OP_ADDR_A, _offset(offset, "a"))


def addr_b(offset: int) -> int:
    return _word(OP_ADDR_B, _offset(offset, "b"))


def addr_c(offset: int) -> int:
    return _word(OP_ADDR_C, _offset(offset, "c"))


def incr(a_inc: int = 0, b_inc: int = 0, c_inc: int = 0) -> int:
    fields = []
    for value, name in ((a_inc, "a_inc"), (b_inc, "b_inc"), (c_inc, "c_inc")):
        if value % INCR_UNIT:
            raise ISAError(f"{name}={value} is not a multiple of {INCR_UNIT}")
        fields.append(_field(value // INCR_UNIT, 16, name))
    return _word(OP_INCR, fields[0] | (fields[1] << 16) | (fields[2] << 32))


def repeat(count: int) -> int:
    if count < 1:
        raise ISAError("repeat count must be at least 1")
    return _word(OP_REPEAT, _field(count, 32, "count"))


def gemm(flags: int = 0) -> int:
    return _word(OP_GEMM, _field(flags, 3, "flags"))


@dataclass(frozen=True)
class Instruction:
    opcode: int
    payload: int

    @property
    def name(self) -> str:
        return NAMES.get(self.opcode, f"op_{self.opcode:02x}")

    def fields(self) -> dict[str, int]:
        p = self.payload
        if self.opcode == OP_SHAPE:
            return {"m": (p >> 24) & 0xFF, "n": (p >> 16) & 0xFF, "k": p & 0xFFFF}
        if self.opcode in (OP_ADDR_A, OP_ADDR_B, OP_ADDR_C):
            return {"offset": p & 0xFFFFFFFF}
        if self.opcode == OP_INCR:
            return {
                "a_inc": (p & 0xFFFF) * INCR_UNIT,
                "b_inc": ((p >> 16) & 0xFFFF) * INCR_UNIT,
                "c_inc": ((p >> 32) & 0xFFFF) * INCR_UNIT,
            }
        if self.opcode == OP_REPEAT:
            return {"count": p & 0xFFFFFFFF}
        if self.opcode == OP_GEMM:
            return {"flags": p & 0x7}
        return {}


def decode(word: int) -> Instruction:
    return Instruction((word >> 56) & 0xFF, word & ((1 << 56) - 1))


def disassemble(words: Iterable[int]) -> str:
    lines = []
    for pc, word in enumerate(words):
        inst = decode(word)
        args = ", ".join(f"{k}={v}" for k, v in inst.fields().items())
        lines.append(f"{pc:6d}: {inst.name:7s} {args}".rstrip())
    return "\n".join(lines)


def pack(words: Iterable[int]) -> bytes:
    return b"".join(struct.pack("<Q", w) for w in words)


def unpack(blob: bytes) -> list[int]:
    if len(blob) % 8:
        raise ISAError("program length is not a multiple of 8 bytes")
    return [w for (w,) in struct.iter_unpack("<Q", blob)]


def align(value: int, to: int = ALIGN) -> int:
    return (value + to - 1) // to * to
