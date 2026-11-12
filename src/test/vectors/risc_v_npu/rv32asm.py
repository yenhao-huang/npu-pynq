"""A small two-pass RV32I + M assembler, so the CVA6 tests need no RISC-V toolchain.

Only uncompressed 32-bit encodings are produced. Programs are built in Python
rather than parsed from text:

    asm = Assembler()
    asm.li("t0", 0x4000_0000)
    asm.label("loop")
    asm.addi("t1", "t1", -1)
    asm.bnez("t1", "loop")
    code = asm.assemble()          # bytes, little-endian

Branch and jump targets are labels (resolved PC-relative, so the result is
position independent) or integer byte offsets. Supported: lui, auipc, jal,
jalr, beq/bne/blt/bge/bltu/bgeu, lb/lh/lw/lbu/lhu, sb/sh/sw, addi/slti/sltiu/
xori/ori/andi/slli/srli/srai, add/sub/sll/slt/sltu/xor/srl/sra/or/and, fence,
ecall, ebreak, mul/mulh/mulhsu/mulhu/div/divu/rem/remu, and the pseudo
instructions li, la, mv, not, neg, nop, j, jr, ret, beqz, bnez, bgt, ble.

`li` always expands to the same number of words for a given value (one word if
it fits a signed 12-bit immediate, else lui [+ addi]), so label addresses are
known after the first pass.
"""

from __future__ import annotations

import struct
from typing import Callable

ABI_NAMES = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1"] + \
    [f"a{i}" for i in range(8)] + [f"s{i}" for i in range(2, 12)] + [f"t{i}" for i in range(3, 7)]
REGISTERS = {name: index for index, name in enumerate(ABI_NAMES)}
REGISTERS.update({f"x{i}": i for i in range(32)})
REGISTERS["fp"] = 8


class AsmError(ValueError):
    """A malformed instruction, operand or label."""


def reg(name: str | int) -> int:
    if isinstance(name, int):
        if 0 <= name < 32:
            return name
        raise AsmError(f"register x{name} does not exist")
    try:
        return REGISTERS[name]
    except KeyError:
        raise AsmError(f"unknown register {name!r}") from None


def _signed(value: int, bits: int, what: str) -> int:
    if not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
        raise AsmError(f"{what} {value} does not fit in {bits} signed bits")
    return value & ((1 << bits) - 1)


# ------------------------------------------------------------------ formats
def r_type(funct7: int, rs2: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def i_type(imm: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    return (_signed(imm, 12, "I immediate") << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def s_type(imm: int, rs2: int, rs1: int, funct3: int, opcode: int) -> int:
    u = _signed(imm, 12, "S immediate")
    return ((u >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((u & 0x1F) << 7) | opcode


def b_type(offset: int, rs2: int, rs1: int, funct3: int, opcode: int = 0x63) -> int:
    if offset % 2:
        raise AsmError(f"branch offset {offset} is odd")
    u = _signed(offset, 13, "branch offset")
    return (((u >> 12) & 1) << 31) | (((u >> 5) & 0x3F) << 25) | (rs2 << 20) | (rs1 << 15) | \
        (funct3 << 12) | (((u >> 1) & 0xF) << 8) | (((u >> 11) & 1) << 7) | opcode


def u_type(imm20: int, rd: int, opcode: int) -> int:
    if not 0 <= imm20 < (1 << 20):
        raise AsmError(f"U immediate {imm20:#x} does not fit in 20 bits")
    return (imm20 << 12) | (rd << 7) | opcode


def j_type(offset: int, rd: int, opcode: int = 0x6F) -> int:
    if offset % 2:
        raise AsmError(f"jump offset {offset} is odd")
    u = _signed(offset, 21, "jump offset")
    return (((u >> 20) & 1) << 31) | (((u >> 1) & 0x3FF) << 21) | (((u >> 11) & 1) << 20) | \
        (((u >> 12) & 0xFF) << 12) | (rd << 7) | opcode


def split_hi_lo(value: int) -> tuple[int, int]:
    """(hi20, lo12) with (hi20 << 12) + sign_extend(lo12) == value mod 2**32."""
    value &= 0xFFFFFFFF
    lo = value & 0xFFF
    if lo >= 0x800:
        lo -= 0x1000
    hi = ((value - lo) >> 12) & 0xFFFFF
    return hi, lo


_BRANCH = {"beq": 0, "bne": 1, "blt": 4, "bge": 5, "bltu": 6, "bgeu": 7}
_LOAD = {"lb": 0, "lh": 1, "lw": 2, "lbu": 4, "lhu": 5}
_STORE = {"sb": 0, "sh": 1, "sw": 2}
_OP_IMM = {"addi": 0, "slti": 2, "sltiu": 3, "xori": 4, "ori": 6, "andi": 7}
_SHIFT_IMM = {"slli": (0x00, 1), "srli": (0x00, 5), "srai": (0x20, 5)}
_OP = {"add": (0x00, 0), "sub": (0x20, 0), "sll": (0x00, 1), "slt": (0x00, 2),
       "sltu": (0x00, 3), "xor": (0x00, 4), "srl": (0x00, 5), "sra": (0x20, 5),
       "or": (0x00, 6), "and": (0x00, 7),
       "mul": (0x01, 0), "mulh": (0x01, 1), "mulhsu": (0x01, 2), "mulhu": (0x01, 3),
       "div": (0x01, 4), "divu": (0x01, 5), "rem": (0x01, 6), "remu": (0x01, 7)}


class Assembler:
    """Collects instructions; `assemble()` resolves labels and returns bytes."""

    def __init__(self) -> None:
        # Each item: a callable (pc, labels) -> list of 32-bit words, and its
        # fixed word count.
        self._items: list[tuple[Callable[[int, dict[str, int]], list[int]], int]] = []
        self._labels: dict[str, int] = {}
        self._pc = 0

    # ------------------------------------------------------------- plumbing
    @property
    def pc(self) -> int:
        """Byte offset of the next instruction."""
        return self._pc

    def _emit(self, fn: Callable[[int, dict[str, int]], list[int]], words: int = 1) -> None:
        self._items.append((fn, words))
        self._pc += 4 * words

    def word(self, value: int) -> None:
        """A raw 32-bit word."""
        self._emit(lambda pc, labels: [value & 0xFFFFFFFF])

    def label(self, name: str) -> None:
        if name in self._labels:
            raise AsmError(f"label {name!r} defined twice")
        self._labels[name] = self._pc

    @staticmethod
    def _target(target: str | int, pc: int, labels: dict[str, int]) -> int:
        if isinstance(target, int):
            return target
        if target not in labels:
            raise AsmError(f"undefined label {target!r}")
        return labels[target] - pc

    def assemble(self) -> bytes:
        words: list[int] = []
        pc = 0
        for fn, count in self._items:
            out = fn(pc, self._labels)
            if len(out) != count:
                raise AsmError("internal: instruction size changed between passes")
            words.extend(out)
            pc += 4 * count
        return b"".join(struct.pack("<I", w) for w in words)

    def words(self) -> list[int]:
        blob = self.assemble()
        return [w for (w,) in struct.iter_unpack("<I", blob)]

    # --------------------------------------------------------- base integer
    def lui(self, rd, imm20: int) -> None:
        self._emit(lambda pc, l: [u_type(imm20, reg(rd), 0x37)])

    def auipc(self, rd, imm20: int) -> None:
        self._emit(lambda pc, l: [u_type(imm20, reg(rd), 0x17)])

    def jal(self, rd, target) -> None:
        self._emit(lambda pc, l: [j_type(self._target(target, pc, l), reg(rd))])

    def jalr(self, rd, rs1, imm: int = 0) -> None:
        self._emit(lambda pc, l: [i_type(imm, reg(rs1), 0, reg(rd), 0x67)])

    def fence(self) -> None:
        self.word(0x0FF0000F)  # fence iorw, iorw

    def ecall(self) -> None:
        self.word(0x00000073)

    def ebreak(self) -> None:
        self.word(0x00100073)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        name = name.rstrip("_")  # and_, or_: Python keywords
        if name in _BRANCH:
            f3 = _BRANCH[name]
            return lambda rs1, rs2, target: self._emit(
                lambda pc, l: [b_type(self._target(target, pc, l), reg(rs2), reg(rs1), f3)])
        if name in _LOAD:
            f3 = _LOAD[name]
            return lambda rd, imm, rs1: self._emit(lambda pc, l: [i_type(imm, reg(rs1), f3, reg(rd), 0x03)])
        if name in _STORE:
            f3 = _STORE[name]
            return lambda rs2, imm, rs1: self._emit(lambda pc, l: [s_type(imm, reg(rs2), reg(rs1), f3, 0x23)])
        if name in _OP_IMM:
            f3 = _OP_IMM[name]
            return lambda rd, rs1, imm: self._emit(lambda pc, l: [i_type(imm, reg(rs1), f3, reg(rd), 0x13)])
        if name in _SHIFT_IMM:
            f7, f3 = _SHIFT_IMM[name]

            def shift(rd, rs1, shamt: int) -> None:
                if not 0 <= shamt < 32:
                    raise AsmError(f"shift amount {shamt} out of range")
                self._emit(lambda pc, l: [r_type(f7, shamt, reg(rs1), f3, reg(rd), 0x13)])
            return shift
        if name in _OP:
            f7, f3 = _OP[name]
            return lambda rd, rs1, rs2: self._emit(
                lambda pc, l: [r_type(f7, reg(rs2), reg(rs1), f3, reg(rd), 0x33)])
        raise AttributeError(name)

    # -------------------------------------------------------------- pseudos
    def nop(self) -> None:
        self.addi("zero", "zero", 0)

    def mv(self, rd, rs) -> None:
        self.addi(rd, rs, 0)

    def not_(self, rd, rs) -> None:
        self.xori(rd, rs, -1)

    def neg(self, rd, rs) -> None:
        self.sub(rd, "zero", rs)

    def li(self, rd, value: int) -> None:
        if not -(1 << 31) <= value < (1 << 32):
            raise AsmError(f"li value {value} does not fit in 32 bits")
        if value >= (1 << 31):
            value -= 1 << 32
        if -2048 <= value < 2048:
            self.addi(rd, "zero", value)
            return
        hi, lo = split_hi_lo(value)
        self.lui(rd, hi)
        if lo:
            self.addi(rd, rd, lo)

    def la(self, rd, target: str) -> None:
        """PC-relative address of a label (auipc + addi)."""
        def first(pc, l):
            hi, _ = split_hi_lo(self._target(target, pc, l))
            return [u_type(hi, reg(rd), 0x17)]

        def second(pc, l):
            _, lo = split_hi_lo(self._target(target, pc - 4, l))
            return [i_type(lo, reg(rd), 0, reg(rd), 0x13)]
        self._emit(first)
        self._emit(second)

    def j(self, target) -> None:
        self.jal("zero", target)

    def jr(self, rs) -> None:
        self.jalr("zero", rs, 0)

    def ret(self) -> None:
        self.jalr("zero", "ra", 0)

    def beqz(self, rs, target) -> None:
        self.beq(rs, "zero", target)

    def bnez(self, rs, target) -> None:
        self.bne(rs, "zero", target)

    def bgt(self, rs1, rs2, target) -> None:
        self.blt(rs2, rs1, target)

    def ble(self, rs1, rs2, target) -> None:
        self.bge(rs2, rs1, target)
