"""Two-pass RV32I assembler for the npu_scalar_core (issue #136).

Syntax, one statement per line::

    label:                      # labels may share a line with a statement
    addi  x1, x0, 5             # comments start with '#', ';' or '//'
    lw    a0, -4(sp)            # loads/stores use imm(rs1)
    beq   t0, t1, loop          # branch/jump operands are target addresses
    .word 0x12345678, label+4   # data words
    .org  0x400                 # advance the location counter (zero fill)
    .equ  NAME, 42              # define a constant

Registers: ``x0``-``x31`` and the ABI names (``zero ra sp gp tp t0-t6 s0-s11
fp a0-a7``). Expressions: integers (decimal, ``0x``, ``0b``, negative),
symbols, ``.`` (address of the current statement), joined with ``+`` / ``-``.
``%hi(expr)`` and ``%lo(expr)`` give the LUI/ADDI split of a 32-bit value.

RV32I: every base instruction, ``fence``, ``fence.i``, ``ecall``, ``ebreak``.

Pseudo-instructions: ``nop``, ``li rd, imm`` (ADDI when the value is a literal
that fits 12 bits, otherwise LUI+ADDI), ``la rd, sym`` (always LUI+ADDI),
``mv``, ``not``, ``neg``, ``seqz``, ``snez``, ``j``, ``jr``, ``call``, ``ret``,
``beqz``, ``bnez``, ``bltz``, ``bgez``, ``blez``, ``bgtz``, ``bgt``, ``ble``,
``bgtu``, ``bleu``; ``jal label`` (rd = ra) and ``jalr rs`` (rd = ra).

Vector instructions (custom-0, opcode 0b0001011). Field packing follows
docs/goals/hexagon_npu/architecture.md: funct7 = {vs2[2:1], op[4:0]},
rs2 = vs1, funct3 = {vs2[0], type[1:0]}, rd = vd, rs1 = scalar register.
``.T`` is the lane type suffix ``.i8`` (0), ``.i16`` (1), ``.i32`` (2) or
``.f16`` (3)::

    vld        vD, xA           op 0   vD <- TCM[x[A]]
    vst        vS, xA           op 1   TCM[x[A]] <- vS      (vs1 = S)
    vsplat.T   vD, xS           op 2   every lane of vD <- x[S]
    vadd.T     vD, vA, vB       op 3   (vs1 = A, vs2 = B)
    vsub.T     vD, vA, vB       op 4
    vmul.T     vD, vA, vB       op 5
    vmax.T     vD, vA, vB       op 6
    vmin.T     vD, vA, vB       op 7
    vrelu.T    vD, vA           op 8
    vsra.T     vD, vA, xS       op 9   shift = x[S][4:0]
    vmov       vD, vA           op 10
    vfence                      op 31  wait until the vector unit is idle
    vcmd       op, type, vD, vA, vB, xS   raw form (any op, e.g. illegal ones)

``vld``/``vst``/``vmov`` accept an optional type suffix (default ``.i8`` = 0).
Operands that an op does not use are encoded as 0 (register x0 / v0).

Library use::

    prog = assemble(source, base=0)
    prog.words        # list of 32-bit ints, prog.words[i] at base + 4 * i
    prog.symbols      # label -> address
    prog.hex()        # one 8-digit hex word per line
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "AsmError",
    "Program",
    "assemble",
    "VECTOR_OPS",
    "VECTOR_TYPES",
    "OPC_CUSTOM0",
    "encode_vector",
    "VFENCE_WORD",
]


class AsmError(ValueError):
    """Assembly failed; the message names the source line."""


ABI_NAMES = {
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4, "t0": 5, "t1": 6, "t2": 7,
    "s0": 8, "fp": 8, "s1": 9, "a0": 10, "a1": 11, "a2": 12, "a3": 13,
    "a4": 14, "a5": 15, "a6": 16, "a7": 17, "s2": 18, "s3": 19, "s4": 20,
    "s5": 21, "s6": 22, "s7": 23, "s8": 24, "s9": 25, "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31,
}

OPC_LUI = 0b0110111
OPC_AUIPC = 0b0010111
OPC_JAL = 0b1101111
OPC_JALR = 0b1100111
OPC_BRANCH = 0b1100011
OPC_LOAD = 0b0000011
OPC_STORE = 0b0100011
OPC_OPIMM = 0b0010011
OPC_OP = 0b0110011
OPC_FENCE = 0b0001111
OPC_SYSTEM = 0b1110011
OPC_CUSTOM0 = 0b0001011

BRANCH_F3 = {"beq": 0, "bne": 1, "blt": 4, "bge": 5, "bltu": 6, "bgeu": 7}
LOAD_F3 = {"lb": 0, "lh": 1, "lw": 2, "lbu": 4, "lhu": 5}
STORE_F3 = {"sb": 0, "sh": 1, "sw": 2}
OPIMM_F3 = {"addi": 0, "slti": 2, "sltiu": 3, "xori": 4, "ori": 6, "andi": 7}
SHIFTIMM = {"slli": (1, 0x00), "srli": (5, 0x00), "srai": (5, 0x20)}
OP_F3F7 = {
    "add": (0, 0x00), "sub": (0, 0x20), "sll": (1, 0x00), "slt": (2, 0x00),
    "sltu": (3, 0x00), "xor": (4, 0x00), "srl": (5, 0x00), "sra": (5, 0x20),
    "or": (6, 0x00), "and": (7, 0x00),
}

VECTOR_OPS = {
    "vld": 0, "vst": 1, "vsplat": 2, "vadd": 3, "vsub": 4, "vmul": 5,
    "vmax": 6, "vmin": 7, "vrelu": 8, "vsra": 9, "vmov": 10, "vfence": 31,
}
VECTOR_TYPES = {"i8": 0, "i16": 1, "i32": 2, "f16": 3}


def _mask32(value: int) -> int:
    return value & 0xFFFFFFFF


def encode_vector(op: int, vtype: int, vd: int, vs1: int, vs2: int, rs1: int) -> int:
    """Encode a custom-0 vector command word."""
    if not (0 <= op < 32 and 0 <= vtype < 4 and 0 <= vd < 8 and 0 <= vs1 < 8
            and 0 <= vs2 < 8 and 0 <= rs1 < 32):
        raise AsmError(f"vector field out of range: {op=} {vtype=} {vd=} {vs1=} {vs2=} {rs1=}")
    funct7 = ((vs2 >> 1) << 5) | op
    funct3 = ((vs2 & 1) << 2) | vtype
    return (funct7 << 25) | (vs1 << 20) | (rs1 << 15) | (funct3 << 12) | (vd << 7) | OPC_CUSTOM0


VFENCE_WORD = encode_vector(31, 0, 0, 0, 0, 0)


def _r(f7, rs2, rs1, f3, rd, opc):
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opc


def _i(imm, rs1, f3, rd, opc):
    if not -2048 <= imm <= 2047:
        raise AsmError(f"I-type immediate {imm} out of range")
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opc


def _s(imm, rs2, rs1, f3, opc):
    if not -2048 <= imm <= 2047:
        raise AsmError(f"S-type immediate {imm} out of range")
    imm &= 0xFFF
    return ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((imm & 0x1F) << 7) | opc


def _b(off, rs2, rs1, f3):
    if off % 2 or not -4096 <= off <= 4094:
        raise AsmError(f"branch offset {off} out of range or odd")
    o = off & 0x1FFF
    return (((o >> 12) & 1) << 31) | (((o >> 5) & 0x3F) << 25) | (rs2 << 20) | (rs1 << 15) \
        | (f3 << 12) | (((o >> 1) & 0xF) << 8) | (((o >> 11) & 1) << 7) | OPC_BRANCH


def _u(imm20, rd, opc):
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opc


def _j(off, rd):
    if off % 2 or not -(1 << 20) <= off < (1 << 20):
        raise AsmError(f"jump offset {off} out of range or odd")
    o = off & 0x1FFFFF
    return (((o >> 20) & 1) << 31) | (((o >> 1) & 0x3FF) << 21) | (((o >> 11) & 1) << 20) \
        | (((o >> 12) & 0xFF) << 12) | (rd << 7) | OPC_JAL


def _hi_lo(value: int) -> tuple[int, int]:
    """Split a 32-bit value into (lui imm20, signed addi imm12)."""
    value = _mask32(value)
    lo = value & 0xFFF
    if lo >= 0x800:
        lo -= 0x1000
    hi = ((value - lo) >> 12) & 0xFFFFF
    return hi, lo


@dataclass
class Program:
    words: list[int]
    symbols: dict[str, int]
    base: int = 0
    lines: list[tuple[int, str]] = field(default_factory=list)  # (address, source)
    padding: set[int] = field(default_factory=set)  # byte addresses of .org fill words

    def hex(self) -> str:
        return "".join(f"{w:08x}\n" for w in self.words)

    def image(self) -> dict[int, int]:
        """Sparse memory image: byte address -> word, without .org fill words."""
        return {self.base + 4 * i: w for i, w in enumerate(self.words)
                if self.base + 4 * i not in self.padding}


class _Assembler:
    def __init__(self, source: str, base: int):
        self.source = source
        self.base = base
        self.symbols: dict[str, int] = {}
        self.final = False

    # ---------------------------------------------------------------- operands
    def reg(self, text: str) -> int:
        t = text.strip().lower()
        if t in ABI_NAMES:
            return ABI_NAMES[t]
        m = re.fullmatch(r"x(\d+)", t)
        if m and int(m.group(1)) < 32:
            return int(m.group(1))
        raise AsmError(f"bad register '{text}'")

    def vreg(self, text: str) -> int:
        m = re.fullmatch(r"v([0-7])", text.strip().lower())
        if not m:
            raise AsmError(f"bad vector register '{text}'")
        return int(m.group(1))

    def expr(self, text: str, pc: int) -> int:
        t = text.strip()
        m = re.fullmatch(r"%(hi|lo)\((.*)\)", t)
        if m:
            hi, lo = _hi_lo(self.expr(m.group(2), pc))
            return hi if m.group(1) == "hi" else lo
        tokens = re.findall(r"[+-]|[^+\-\s]+", t)
        if not tokens:
            raise AsmError("empty expression")
        total, sign, expect_term = 0, 1, True
        for tok in tokens:
            if tok in "+-":
                if expect_term:
                    sign = -sign if tok == "-" else sign   # unary sign
                else:
                    sign = -1 if tok == "-" else 1
                    expect_term = True
                continue
            if not expect_term:
                raise AsmError(f"bad expression '{text}'")
            total += sign * self.term(tok, pc)
            sign, expect_term = 1, False
        if expect_term:
            raise AsmError(f"bad expression '{text}'")
        return total

    def term(self, term: str, pc: int) -> int:
        if term == ".":
            return pc
        if re.fullmatch(r"0[xX][0-9a-fA-F_]+|0[bB][01_]+|\d+", term):
            return int(term.replace("_", ""), 0)
        if re.fullmatch(r"[A-Za-z_.$][\w.$]*", term):
            if term in self.symbols:
                return self.symbols[term]
            if self.final:
                raise AsmError(f"undefined symbol '{term}'")
            return 0
        raise AsmError(f"bad term '{term}'")

    def is_literal(self, text: str) -> bool:
        return re.fullmatch(r"\s*-?\s*(0[xX][0-9a-fA-F_]+|0[bB][01_]+|\d+)\s*", text) is not None

    def mem_operand(self, text: str, pc: int) -> tuple[int, int]:
        m = re.fullmatch(r"(.*)\(\s*([\w]+)\s*\)", text.strip())
        if not m:
            raise AsmError(f"bad memory operand '{text}'")
        off = m.group(1).strip()
        return (self.expr(off, pc) if off else 0), self.reg(m.group(2))

    # ---------------------------------------------------------------- statements
    def statement(self, mnem: str, ops: list[str], pc: int) -> list[int]:
        """Encode one statement at address pc; returns its words."""
        n = len(ops)

        def want(count):
            if n != count:
                raise AsmError(f"'{mnem}' takes {count} operands, got {n}")

        def branch(f3, rs1, rs2, target):
            return _b(self.expr(target, pc) - pc, self.reg(rs2), self.reg(rs1), f3)

        if mnem in BRANCH_F3:
            want(3)
            return [branch(BRANCH_F3[mnem], ops[0], ops[1], ops[2])]
        swapped = {"bgt": "blt", "ble": "bge", "bgtu": "bltu", "bleu": "bgeu"}
        if mnem in swapped:
            want(3)
            return [branch(BRANCH_F3[swapped[mnem]], ops[1], ops[0], ops[2])]
        zero_cmp = {"beqz": ("beq", False), "bnez": ("bne", False), "bltz": ("blt", False),
                    "bgez": ("bge", False), "blez": ("bge", True), "bgtz": ("blt", True)}
        if mnem in zero_cmp:
            want(2)
            base, swap = zero_cmp[mnem]
            a, b = ("zero", ops[0]) if swap else (ops[0], "zero")
            return [branch(BRANCH_F3[base], a, b, ops[1])]
        if mnem in LOAD_F3:
            want(2)
            off, rs1 = self.mem_operand(ops[1], pc)
            return [_i(off, rs1, LOAD_F3[mnem], self.reg(ops[0]), OPC_LOAD)]
        if mnem in STORE_F3:
            want(2)
            off, rs1 = self.mem_operand(ops[1], pc)
            return [_s(off, self.reg(ops[0]), rs1, STORE_F3[mnem], OPC_STORE)]
        if mnem in OPIMM_F3:
            want(3)
            return [_i(self.expr(ops[2], pc), self.reg(ops[1]), OPIMM_F3[mnem],
                       self.reg(ops[0]), OPC_OPIMM)]
        if mnem in SHIFTIMM:
            want(3)
            f3, f7 = SHIFTIMM[mnem]
            sh = self.expr(ops[2], pc)
            if not 0 <= sh <= 31:
                raise AsmError(f"shift amount {sh} out of range")
            return [_r(f7, sh, self.reg(ops[1]), f3, self.reg(ops[0]), OPC_OPIMM)]
        if mnem in OP_F3F7:
            want(3)
            f3, f7 = OP_F3F7[mnem]
            return [_r(f7, self.reg(ops[2]), self.reg(ops[1]), f3, self.reg(ops[0]), OPC_OP)]
        if mnem in ("lui", "auipc"):
            want(2)
            imm = self.expr(ops[1], pc)
            if not -(1 << 19) <= imm < (1 << 20):
                raise AsmError(f"U-type immediate {imm} out of range")
            return [_u(imm, self.reg(ops[0]), OPC_LUI if mnem == "lui" else OPC_AUIPC)]
        if mnem == "jal":
            if n == 1:
                return [_j(self.expr(ops[0], pc) - pc, 1)]
            want(2)
            return [_j(self.expr(ops[1], pc) - pc, self.reg(ops[0]))]
        if mnem == "jalr":
            if n == 1:
                return [_i(0, self.reg(ops[0]), 0, 1, OPC_JALR)]
            if n == 2:
                if "(" not in ops[1]:
                    return [_i(0, self.reg(ops[1]), 0, self.reg(ops[0]), OPC_JALR)]
                off, rs1 = self.mem_operand(ops[1], pc)
                return [_i(off, rs1, 0, self.reg(ops[0]), OPC_JALR)]
            want(3)
            return [_i(self.expr(ops[2], pc), self.reg(ops[1]), 0, self.reg(ops[0]), OPC_JALR)]
        if mnem == "fence":
            return [0x0FF0000F]
        if mnem == "fence.i":
            want(0)
            return [0x0000100F]
        if mnem == "ecall":
            want(0)
            return [0x00000073]
        if mnem == "ebreak":
            want(0)
            return [0x00100073]
        # ---------------------------------------------------------- pseudo
        if mnem == "nop":
            want(0)
            return [_i(0, 0, 0, 0, OPC_OPIMM)]
        if mnem == "li":
            want(2)
            rd = self.reg(ops[0])
            value = self.expr(ops[1], pc)
            if not -(1 << 31) <= value < (1 << 32):
                raise AsmError(f"li value {value} out of 32-bit range")
            if self.is_literal(ops[1]) and -2048 <= value <= 2047:
                return [_i(value, 0, 0, rd, OPC_OPIMM)]
            hi, lo = _hi_lo(value)
            return [_u(hi, rd, OPC_LUI), _i(lo, rd, 0, rd, OPC_OPIMM)]
        if mnem == "la":
            want(2)
            rd = self.reg(ops[0])
            hi, lo = _hi_lo(self.expr(ops[1], pc))
            return [_u(hi, rd, OPC_LUI), _i(lo, rd, 0, rd, OPC_OPIMM)]
        if mnem == "mv":
            want(2)
            return [_i(0, self.reg(ops[1]), 0, self.reg(ops[0]), OPC_OPIMM)]
        if mnem == "not":
            want(2)
            return [_i(-1, self.reg(ops[1]), 4, self.reg(ops[0]), OPC_OPIMM)]
        if mnem == "neg":
            want(2)
            return [_r(0x20, self.reg(ops[1]), 0, 0, self.reg(ops[0]), OPC_OP)]
        if mnem == "seqz":
            want(2)
            return [_i(1, self.reg(ops[1]), 3, self.reg(ops[0]), OPC_OPIMM)]
        if mnem == "snez":
            want(2)
            return [_r(0, self.reg(ops[1]), 0, 3, self.reg(ops[0]), OPC_OP)]
        if mnem == "j":
            want(1)
            return [_j(self.expr(ops[0], pc) - pc, 0)]
        if mnem == "call":
            want(1)
            return [_j(self.expr(ops[0], pc) - pc, 1)]
        if mnem == "jr":
            want(1)
            return [_i(0, self.reg(ops[0]), 0, 0, OPC_JALR)]
        if mnem == "ret":
            want(0)
            return [_i(0, 1, 0, 0, OPC_JALR)]
        # ---------------------------------------------------------- vector
        return [self.vector(mnem, ops, pc)]

    def vector(self, mnem: str, ops: list[str], pc: int) -> int:
        base, _, suffix = mnem.partition(".")
        if base == "vcmd":
            if len(ops) != 6:
                raise AsmError("'vcmd' takes op, type, vd, vs1, vs2, xS")
            return encode_vector(self.expr(ops[0], pc), self.expr(ops[1], pc), self.vreg(ops[2]),
                                 self.vreg(ops[3]), self.vreg(ops[4]), self.reg(ops[5]))
        if base not in VECTOR_OPS:
            raise AsmError(f"unknown mnemonic '{mnem}'")
        op = VECTOR_OPS[base]
        if suffix:
            if suffix not in VECTOR_TYPES:
                raise AsmError(f"unknown vector type '.{suffix}'")
            vtype = VECTOR_TYPES[suffix]
        elif base in ("vld", "vst", "vmov", "vfence"):
            vtype = 0
        else:
            raise AsmError(f"'{base}' needs a type suffix (.i8 .i16 .i32 .f16)")
        shapes = {
            "vld": "dx", "vst": "ax", "vsplat": "dx", "vadd": "dab", "vsub": "dab",
            "vmul": "dab", "vmax": "dab", "vmin": "dab", "vrelu": "da", "vsra": "dax",
            "vmov": "da", "vfence": "",
        }
        shape = shapes[base]
        if len(ops) != len(shape):
            raise AsmError(f"'{mnem}' takes {len(shape)} operands, got {len(ops)}")
        fields = {"d": 0, "a": 0, "b": 0, "x": 0}
        for kind, text in zip(shape, ops):
            fields[kind] = self.reg(text) if kind == "x" else self.vreg(text)
        return encode_vector(op, vtype, fields["d"], fields["a"], fields["b"], fields["x"])

    # ---------------------------------------------------------------- passes
    def size_of(self, mnem: str, ops: list[str], pc: int) -> int:
        if mnem == "la":
            return 2
        if mnem == "li":
            if len(ops) == 2 and self.is_literal(ops[1]):
                return 1 if -2048 <= int(ops[1].replace(" ", "").replace("_", ""), 0) <= 2047 else 2
            return 2
        return 1

    def parse(self):
        stmts = []
        for lineno, raw in enumerate(self.source.splitlines(), 1):
            line = re.split(r"#|;|//", raw, maxsplit=1)[0].strip()
            while True:
                m = re.match(r"([A-Za-z_.$][\w.$]*)\s*:(.*)$", line)
                if not m:
                    break
                stmts.append((lineno, raw, "label", m.group(1), []))
                line = m.group(2).strip()
            if not line:
                continue
            parts = line.split(None, 1)
            mnem = parts[0].lower()
            ops = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
            if ops == [""]:
                ops = []
            stmts.append((lineno, raw, "stmt", mnem, ops))
        return stmts

    def run(self) -> Program:
        stmts = self.parse()
        # Pass 1: addresses.
        for final in (False, True):
            self.final = final
            pc = self.base
            words: list[int] = []
            lines: list[tuple[int, str]] = []
            padding: set[int] = set()
            for lineno, raw, kind, name, ops in stmts:
                try:
                    if kind == "label":
                        if not final:
                            if name in self.symbols:
                                raise AsmError(f"duplicate label '{name}'")
                            self.symbols[name] = pc
                        continue
                    if name == ".equ":
                        if len(ops) != 2:
                            raise AsmError(".equ takes NAME, value")
                        self.symbols[ops[0]] = self.expr(ops[1], pc)
                        continue
                    if name == ".org":
                        target = self.expr(ops[0], pc)
                        if target < pc or target % 4:
                            raise AsmError(f".org {target:#x} moves backwards or is unaligned")
                        while pc < target:
                            words.append(0)
                            padding.add(pc)
                            pc += 4
                        continue
                    if name == ".word":
                        for op in ops:
                            words.append(_mask32(self.expr(op, pc)) if final else 0)
                            lines.append((pc, raw.strip()))
                            pc += 4
                        continue
                    if name.startswith("."):
                        raise AsmError(f"unknown directive '{name}'")
                    if final:
                        enc = self.statement(name, ops, pc)
                        if len(enc) != self.size_of(name, ops, pc):
                            raise AsmError("internal size mismatch")
                    else:
                        enc = [0] * self.size_of(name, ops, pc)
                    for w in enc:
                        words.append(_mask32(w))
                        lines.append((pc, raw.strip()))
                        pc += 4
                except AsmError as exc:
                    raise AsmError(f"line {lineno}: {exc}: {raw.strip()}") from None
                except (ValueError, KeyError) as exc:
                    raise AsmError(f"line {lineno}: {exc}: {raw.strip()}") from None
        return Program(words, dict(self.symbols), self.base, lines, padding)


def assemble(source: str, base: int = 0) -> Program:
    """Assemble source text placed at byte address base (4-aligned)."""
    if base % 4:
        raise AsmError("base address must be 4-aligned")
    return _Assembler(source, base).run()


def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("-o", "--output", help="hex output file (default stdout)")
    parser.add_argument("--base", type=lambda s: int(s, 0), default=0)
    args = parser.parse_args(argv)
    with open(args.source, encoding="utf-8") as handle:
        prog = assemble(handle.read(), args.base)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(prog.hex())
    else:
        sys.stdout.write(prog.hex())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
