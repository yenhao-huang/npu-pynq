"""Reference model of the Hexagon-style vector unit (``npu_vector_unit``, issue #136).

The model is architectural: it executes one command at a time, in order, on a
register file of ``nregs`` VLEN-bit registers and a TCM of VLEN-bit words.
The RTL is pipelined but stalls on every hazard, so its final register file,
TCM image and ``illegal`` flag must equal this model's after any command
sequence.

Lane arithmetic uses the golden numeric helpers in :mod:`hx_numeric`:

* integer ADD/SUB saturate, MUL keeps the low half, MAX/MIN are signed,
  RELU is ``max(x, 0)``, SRA is ``sra_round(x, scalar[4:0], width)`` (so a
  shift of at least the lane width gives 0, a shift of 0 is a copy);
* FP16 ADD/SUB/MUL are ``fp16_add``/``fp16_mul``; MAX/MIN are
  ``fp16_max``/``fp16_min``; RELU is ``fp16_max(x, +0)``, so
  ``relu(-0) = +0`` and ``relu(NaN) = 0x7E00``.

Illegal commands (opcode > 10, or VSRA with type FP16) change nothing except
the sticky ``illegal`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.test.model import hx_numeric as hx

OP_VLD = 0
OP_VST = 1
OP_VSPLAT = 2
OP_VADD = 3
OP_VSUB = 4
OP_VMUL = 5
OP_VMAX = 6
OP_VMIN = 7
OP_VRELU = 8
OP_VSRA = 9
OP_VMOV = 10

OP_NAMES = {
    OP_VLD: "VLD",
    OP_VST: "VST",
    OP_VSPLAT: "VSPLAT",
    OP_VADD: "VADD",
    OP_VSUB: "VSUB",
    OP_VMUL: "VMUL",
    OP_VMAX: "VMAX",
    OP_VMIN: "VMIN",
    OP_VRELU: "VRELU",
    OP_VSRA: "VSRA",
    OP_VMOV: "VMOV",
}

TYPE_INT8 = 0
TYPE_INT16 = 1
TYPE_INT32 = 2
TYPE_FP16 = 3

TYPE_NAMES = {TYPE_INT8: "INT8", TYPE_INT16: "INT16", TYPE_INT32: "INT32", TYPE_FP16: "FP16"}
LANE_BITS = {TYPE_INT8: 8, TYPE_INT16: 16, TYPE_INT32: 32, TYPE_FP16: 16}


@dataclass(frozen=True)
class VectorCommand:
    op: int
    typ: int = 0
    vd: int = 0
    vs1: int = 0
    vs2: int = 0
    addr: int = 0
    scalar: int = 0


def is_legal(op: int, typ: int) -> bool:
    if op > OP_VMOV:
        return False
    return not (op == OP_VSRA and typ == TYPE_FP16)


def split_lanes(word: int, bits: int, vlen: int) -> List[int]:
    mask = (1 << bits) - 1
    return [(word >> (i * bits)) & mask for i in range(vlen // bits)]


def join_lanes(lanes: List[int], bits: int) -> int:
    mask = (1 << bits) - 1
    word = 0
    for i, lane in enumerate(lanes):
        word |= (lane & mask) << (i * bits)
    return word


def fp16_relu(x: int) -> int:
    """relu(x) = fp16_max(x, +0): relu(-0) = +0, relu(NaN) = 0x7E00."""
    return hx.fp16_max(x, 0x0000)


def int_lane_op(op: int, a: int, b: int, bits: int, scalar: int) -> int:
    """One signed integer lane; a and b are raw lane bits, result raw bits."""
    sa = hx.to_signed(a, bits)
    sb = hx.to_signed(b, bits)
    if op == OP_VADD:
        r = hx.int_sat(sa + sb, bits)
    elif op == OP_VSUB:
        r = hx.int_sat(sa - sb, bits)
    elif op == OP_VMUL:
        r = hx.int_wrap(sa * sb, bits)
    elif op == OP_VMAX:
        r = max(sa, sb)
    elif op == OP_VMIN:
        r = min(sa, sb)
    elif op == OP_VRELU:
        r = max(sa, 0)
    elif op == OP_VSRA:
        r = hx.sra_round(sa, scalar & 0x1F, bits)
    else:
        raise ValueError(f"not an integer lane op: {op}")
    return r & ((1 << bits) - 1)


def fp16_lane_op(op: int, a: int, b: int) -> int:
    if op == OP_VADD:
        return hx.fp16_add(a, b)
    if op == OP_VSUB:
        return hx.fp16_add(a, b, sub=True)
    if op == OP_VMUL:
        return hx.fp16_mul(a, b)
    if op == OP_VMAX:
        return hx.fp16_max(a, b)
    if op == OP_VMIN:
        return hx.fp16_min(a, b)
    if op == OP_VRELU:
        return fp16_relu(a)
    raise ValueError(f"not an FP16 lane op: {op}")


def splat(typ: int, scalar: int, vlen: int) -> int:
    bits = LANE_BITS[typ]
    return join_lanes([scalar & ((1 << bits) - 1)] * (vlen // bits), bits)


def vector_op(op: int, typ: int, a: int, b: int, scalar: int, vlen: int = 64) -> int:
    """Lane-wise arithmetic op on two VLEN-bit words (VADD..VSRA)."""
    bits = LANE_BITS[typ]
    la = split_lanes(a, bits, vlen)
    lb = split_lanes(b, bits, vlen)
    if typ == TYPE_FP16:
        out = [fp16_lane_op(op, x, y) for x, y in zip(la, lb)]
    else:
        out = [int_lane_op(op, x, y, bits, scalar) for x, y in zip(la, lb)]
    return join_lanes(out, bits)


@dataclass
class VectorUnitModel:
    """Architectural state of npu_vector_unit: registers, TCM, illegal flag."""

    vlen: int = 64
    nregs: int = 8
    tcm_words: int = 1024
    rf: List[int] = field(default_factory=list)
    tcm: List[int] = field(default_factory=list)
    illegal: bool = False

    def __post_init__(self) -> None:
        if not self.rf:
            self.rf = [0] * self.nregs
        if not self.tcm:
            self.tcm = [0] * self.tcm_words

    @property
    def mask(self) -> int:
        return (1 << self.vlen) - 1

    def execute(self, cmd: VectorCommand) -> Optional[int]:
        """Execute one command; returns the value written (None if none)."""
        op, typ = cmd.op & 0x1F, cmd.typ & 3
        if not is_legal(op, typ):
            self.illegal = True
            return None
        addr = cmd.addr % self.tcm_words
        if op == OP_VLD:
            value = self.tcm[addr]
        elif op == OP_VST:
            self.tcm[addr] = self.rf[cmd.vs1]
            return self.tcm[addr]
        elif op == OP_VSPLAT:
            value = splat(typ, cmd.scalar, self.vlen)
        elif op == OP_VMOV:
            value = self.rf[cmd.vs1]
        else:
            value = vector_op(op, typ, self.rf[cmd.vs1], self.rf[cmd.vs2], cmd.scalar, self.vlen)
        self.rf[cmd.vd] = value & self.mask
        return self.rf[cmd.vd]

    def run(self, commands) -> None:
        for cmd in commands:
            self.execute(cmd)


__all__ = [
    "OP_VLD", "OP_VST", "OP_VSPLAT", "OP_VADD", "OP_VSUB", "OP_VMUL", "OP_VMAX",
    "OP_VMIN", "OP_VRELU", "OP_VSRA", "OP_VMOV", "OP_NAMES",
    "TYPE_INT8", "TYPE_INT16", "TYPE_INT32", "TYPE_FP16", "TYPE_NAMES", "LANE_BITS",
    "VectorCommand", "VectorUnitModel", "is_legal", "vector_op", "splat",
    "fp16_relu", "split_lanes", "join_lanes",
]
