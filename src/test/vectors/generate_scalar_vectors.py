"""Generate the program vectors for tb_npu_scalar_core (issue #136).

Every case is a set of assembly programs placed in one 64 KB memory image,
assembled with src/test/model/rv32_asm.py and executed by the reference ISS
src/test/model/rv32_iss.py. A case has one or more runs; each run is a
`start` at some PC that ends in a halt or a fault. Registers persist across
the runs of a case; memory is reloaded per case.

Output file (whitespace separated, hex unless noted)::

    # comment line
    CASE <name> <i_latency (dec)> <d_latency (dec)> <n_init>
    <addr> <word>                                   n_init lines
    RUNS <n (dec)>
    RUN <start_pc> <status 1=halt 2=fault (dec)> <cause (dec)> <fault_pc>
        <instret (dec)> <cycles (dec, -1 = not checked)>
    <x1> ... <x31>                                  one per line
    VCMDS <m (dec)>
    <op> <type> <vd> <vs1> <vs2> <scalar>           m lines
    MEM <p (dec)>
    <addr> <word>                                   every initialised or written word
    ...
    DONE

Latencies -1 run the case with random ready / response latency (0..4 cycles)
and a random vector unit in the testbench. Latencies >= 0 run it with both
ports always ready, those fixed response latencies (0 = response in the
handshake cycle), vcmd_ready high and vec_busy low, and the cycle counter is
checked against the ISS cycle model (CPI measurements) unless cycles is -1
(the model needs equal latencies).

    python3 -m src.test.vectors.generate_scalar_vectors [--check]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from src.test.model.rv32_asm import assemble
from src.test.model.rv32_iss import RV32ISS

OUTPUT = Path(__file__).with_name("scalar_core_programs.txt")
SEED = 0x136
MEM_SIZE = 0x10000
DATA = 0x8000      # data region
SIG = 0xA000       # signature region of the coverage program
SCRATCH = 0xC000   # random-program load/store region

PROLOGUE = "\n".join(f"    addi x{i}, x0, 0" for i in range(1, 32)) + "\n"

INTERESTING = [0, 1, -1, 0x7FFFFFFF, -0x80000000, 0x12345678, -0x01234568]


class Builder:
    """Accumulates assembly with unique labels and a signature store helper."""

    def __init__(self, sig_base_reg="s0"):
        self.lines: list[str] = []
        self.sig_reg = sig_base_reg
        self.sig_off = 0
        self.n_labels = 0

    def __call__(self, *lines: str):
        self.lines.extend(lines)

    def label(self) -> str:
        self.n_labels += 1
        return f"L{self.n_labels}"

    def sig(self, reg: str):
        if self.sig_off >= 2040:
            self(f"    addi {self.sig_reg}, {self.sig_reg}, 2040")
            self.sig_off = 0
        self(f"    sw {reg}, {self.sig_off}({self.sig_reg})")
        self.sig_off += 4

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _h(v: int) -> str:
    return hex(v & 0xFFFFFFFF)


# --------------------------------------------------------------------- (a)
def coverage_program() -> str:
    b = Builder()
    b(PROLOGUE, f"    li s0, {SIG}", f"    li s1, {DATA}")
    # LUI / AUIPC
    for imm in (0x80000, 0xFFFFF, 0, 0x12345):
        b(f"    lui t0, {imm:#x}")
        b.sig("t0")
    for imm in (0, 0xFFFFF, 1, 0x80000):
        b(f"    auipc t0, {imm:#x}")
        b.sig("t0")
    # OP-IMM
    imms = [0, 1, -1, 2047, -2048, 0x555]
    for op in ("addi", "slti", "sltiu", "xori", "ori", "andi"):
        for v in INTERESTING:
            b(f"    li t0, {_h(v)}")
            for imm in imms:
                b(f"    {op} t1, t0, {imm}")
                b.sig("t1")
    for op in ("slli", "srli", "srai"):
        for v in INTERESTING:
            b(f"    li t0, {_h(v)}")
            for sh in (0, 1, 4, 31):
                b(f"    {op} t1, t0, {sh}")
                b.sig("t1")
    # OP: second operands include shift amounts with upper bits set.
    seconds = [0, 1, -1, 0x7FFFFFFF, -0x80000000, 31, 32, 0x23]
    for op in ("add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and"):
        for v in INTERESTING:
            b(f"    li t0, {_h(v)}")
            for w in seconds:
                b(f"    li t1, {_h(w)}")
                b(f"    {op} t2, t0, t1")
                b.sig("t2")
    # rd == rs1 == rs2
    b("    li t0, 0x0f0f0f0f", "    add t0, t0, t0")
    b.sig("t0")
    b("    sub t0, t0, t0")
    b.sig("t0")
    # Branches: all conditions, taken and not taken (t2 = 0 taken, 1 not taken).
    bvals = [0, 1, -1, 0x7FFFFFFF, -0x80000000]
    for op in ("beq", "bne", "blt", "bge", "bltu", "bgeu"):
        for v in bvals:
            for w in bvals:
                lab = b.label()
                b(f"    li t0, {_h(v)}", f"    li t1, {_h(w)}", "    li t2, 0",
                  f"    {op} t0, t1, {lab}", "    li t2, 1", f"{lab}:")
                b.sig("t2")
        lab = b.label()
        b("    li t2, 0", f"    {op} t0, t0, {lab}", "    li t2, 1", f"{lab}:")
        b.sig("t2")
    # Operands that differ in a single bit, for every bit and every condition
    # (one bit per condition in t2: 1 = not taken).
    for v in (0x5A5A5A5A, 0xA5A5A5A5):
        b(f"    li t0, {_h(v)}")
        for k in range(32):
            b(f"    li t1, {_h(v ^ (1 << k))}", "    li t2, 0")
            for op in ("beq", "bne", "blt", "bge", "bltu", "bgeu"):
                lab = b.label()
                b("    slli t2, t2, 1", f"    {op} t0, t1, {lab}", "    ori t2, t2, 1", f"{lab}:")
            b.sig("t2")
    # Backward branch (taken) and a loop counting down.
    b("    li t0, 5", "    li t1, 0")
    loop = b.label()
    b(f"{loop}:", "    addi t1, t1, 3", "    addi t0, t0, -1", f"    bnez t0, {loop}")
    b.sig("t1")
    # Loads: sign / zero extension for every lane.
    words = [0x80817F01, 0x00FFFF00, 0x7FFF8000, 0xDEADBEEF]
    for k in range(len(words)):
        base = 4 * k
        for off in range(4):
            for op in ("lb", "lbu"):
                b(f"    {op} t1, {base + off}(s1)")
                b.sig("t1")
        for off in (0, 2):
            for op in ("lh", "lhu"):
                b(f"    {op} t1, {base + off}(s1)")
                b.sig("t1")
        b(f"    lw t1, {base}(s1)")
        b.sig("t1")
    b("    addi t2, s1, 16", "    lw t1, -4(t2)")
    b.sig("t1")
    b("    lb t1, -1(t2)")
    b.sig("t1")
    # Stores: byte and half lanes over 0x00000000 and 0xFFFFFFFF words.
    off = 0x100
    for fill in (0, -1):
        for lane in range(4):
            b(f"    li t0, {fill}", f"    sw t0, {off}(s1)", f"    li t1, {_h(0x12345678 + 0x11 * lane)}",
              f"    sb t1, {off + lane}(s1)", f"    lw t2, {off}(s1)")
            b.sig("t2")
            off += 4
        for lane in (0, 2):
            b(f"    li t0, {fill}", f"    sw t0, {off}(s1)", f"    li t1, {_h(0x89ABCDEF + lane)}",
              f"    sh t1, {off + lane}(s1)", f"    lw t2, {off}(s1)")
            b.sig("t2")
            off += 4
    b("    li t1, 0xcafef00d", f"    sw t1, {off}(s1)", f"    lw t2, {off}(s1)")
    b.sig("t2")
    b(f"    addi t2, s1, {off + 64}", "    sw t1, -60(t2)", "    sb t1, -53(t2)", "    sh t1, -50(t2)")
    # JAL / JALR with rd = x0, ra, others.
    l1, l2, l3, l4, l5, l6, l7, l8 = (b.label() for _ in range(8))
    b(f"    jal ra, {l1}", "    li t0, 99", f"{l1}:")
    b.sig("ra")
    b("    li t0, 7", f"    jal x0, {l2}", "    li t0, 99", f"{l2}:")
    b.sig("t0")
    b(f"    jal t3, {l3}", "    li t3, 99", f"{l3}:")
    b.sig("t3")
    b(f"    la t0, {l4}", "    jalr ra, 0(t0)", "    li ra, 99", f"{l4}:")
    b.sig("ra")
    b(f"    la t0, {l5}", "    li t1, 5", "    jalr x0, t0, 0", "    li t1, 99", f"{l5}:")
    b.sig("t1")
    b(f"    la t0, {l6}", "    jalr ra, 1(t0)", "    li ra, 99", f"{l6}:")   # bit 0 cleared
    b.sig("ra")
    b(f"    la t0, {l7}", "    jalr t0, 0(t0)", "    li t0, 99", f"{l7}:")   # rd == rs1
    b.sig("t0")
    b(f"    la t0, {l8}+8", "    jalr ra, -8(t0)", "    li ra, 99", f"{l8}:")
    b.sig("ra")
    fwd, back, after, func = (b.label() for _ in range(4))
    b(f"    j {fwd}", f"{back}:", "    li t1, 77", f"    j {after}", f"{fwd}:", f"    j {back}", f"{after}:")
    b.sig("t1")
    b("    li a0, 40", f"    call {func}")
    b.sig("a0")
    # Writes to x0 are dropped.
    nxt = b.label()
    b("    addi x0, x0, 5", "    lui x0, 1", "    lw x0, 0(s1)", "    add x0, s1, s1",
      "    auipc x0, 3", f"    jal x0, {nxt}", f"{nxt}:", "    lbu x0, 3(s1)")
    b.sig("x0")
    b("    addi t1, x0, 0")
    b.sig("t1")
    # FENCE / FENCE.I are NOPs.
    b("    li t1, 3", "    fence", "    fence.i", "    nop", "    addi t1, t1, 1")
    b.sig("t1")
    b("    ecall")
    b(f"{func}:", "    addi a0, a0, 2", "    ret")
    b(f".org {DATA}")
    b("    .word " + ", ".join(hex(w) for w in words))
    return b.text()


# --------------------------------------------------------------------- (b)
def memcpy_program(rng) -> str:
    src = [rng.getrandbits(32) for _ in range(48)]
    return PROLOGUE + f"""
    li a0, {DATA + 0x400}       # word copy, 32 words
    li a1, {DATA}
    li a2, 32
word_loop:
    lw t0, 0(a1)
    sw t0, 0(a0)
    addi a0, a0, 4
    addi a1, a1, 4
    addi a2, a2, -1
    bnez a2, word_loop
    li a0, {DATA + 0x603}       # byte copy, misaligned source and destination
    li a1, {DATA + 0x81}
    li a2, 29
byte_loop:
    lbu t0, 0(a1)
    sb t0, 0(a0)
    addi a0, a0, 1
    addi a1, a1, 1
    addi a2, a2, -1
    bnez a2, byte_loop
    ecall
.org {DATA}
    .word {", ".join(hex(w) for w in src)}
"""


def bubble_program(rng) -> str:
    values = [rng.randint(-1000, 1000) for _ in range(14)] + [0x7FFFFFFF, -0x80000000]
    rng.shuffle(values)
    return PROLOGUE + f"""
    li s2, 16
outer:
    addi s2, s2, -1
    blez s2, done
    li a0, {DATA}
    mv a1, s2
inner:
    lw t0, 0(a0)
    lw t1, 4(a0)
    ble t0, t1, noswap
    sw t1, 0(a0)
    sw t0, 4(a0)
noswap:
    addi a0, a0, 4
    addi a1, a1, -1
    bnez a1, inner
    j outer
done:
    ecall
.org {DATA}
    .word {", ".join(_h(v) for v in values)}
"""


def fib_program() -> str:
    return PROLOGUE + f"""
    li sp, {MEM_SIZE}
    li s1, {DATA}
    li a0, 10
    call fib
    sw a0, 0(s1)
    li a0, 1
    call fib
    sw a0, 4(s1)
    ecall
fib:                        # a0 = fib(a0), recursive, uses the stack
    addi sp, sp, -12
    sw ra, 8(sp)
    sw s0, 4(sp)
    sw a0, 0(sp)
    li t0, 2
    blt a0, t0, fib_ret
    addi a0, a0, -1
    call fib
    mv s0, a0
    lw a0, 0(sp)
    addi a0, a0, -2
    call fib
    add a0, a0, s0
fib_ret:
    lw ra, 8(sp)
    lw s0, 4(sp)
    addi sp, sp, 12
    ret
"""


# --------------------------------------------------------------------- (c)
def vector_program() -> str:
    return PROLOGUE + f"""
    li t0, 0x10
    li t1, 0x20
    li t2, -5
    li t3, 0x23                 # shift operand with upper bits set
    vld v1, t0
    vld.i16 v2, t1
    vsplat.i16 v0, t2
    vadd.i8 v3, v1, v2
    vsub.i16 v4, v3, v0
    vmul.i32 v5, v4, v7
    vmax.f16 v6, v5, v6
    vmin.i8 v7, v6, v5
    vrelu.i16 v2, v7
    vsra.i32 v1, v1, t3
    vmov v0, v7
    vfence
    li s1, {DATA}
    sw t2, 0(s1)                # scalar work after the fence
    vst v3, t1
    vcmd 15, 3, v7, v7, v7, t2  # illegal op: the core forwards it unchanged
    vcmd 0, 3, v5, v6, v3, x0
    vfence
    vfence
    li a0, 100
    li a1, 0x40
    li a2, 8
vloop:
    vsplat.i32 v1, a0
    vadd.i32 v2, v1, v2
    vst v2, a1
    sw a0, 4(s1)
    addi s1, s1, 4
    addi a0, a0, -13
    addi a1, a1, 1
    addi a2, a2, -1
    bnez a2, vloop
    vfence
    vsra.i8 v4, v4, a0
    ecall
"""


# --------------------------------------------------------------------- (d)
FAULT_ENTRIES = [
    ("f_csr", "    li ra, 0x1234\n    .word 0x34011073\n"),            # csrrw: illegal
    ("f_zero", "    .word 0\n"),
    ("f_mul", "    .word 0x02208033\n"),                              # mul (M ext.)
    ("f_slli5", "    .word 0x02009093\n"),                            # slli with shamt[5]
    ("f_mret", "    .word 0x30200073\n"),
    ("f_ld", "    .word 0x0000b083\n"),                               # ld (RV64)
    ("f_jalr_f3", "    .word 0x000090e7\n"),                          # jalr funct3=1
    ("f_branch_f3", "    .word 0x00002063\n"),                        # branch funct3=2
    ("f_store_f3", "    .word 0x00003023\n"),                         # sd (RV64)
    ("f_addi_ok", "    .word 0x40000093\n"),                          # addi x1, x0, 0x400: legal
    ("f_op_f7", "    .word 0x40001033\n"),                            # sll with funct7=0x20
    ("f_fence_f3", "    .word 0x0000200f\n"),
    ("f_custom1", "    .word 0x0000002b\n"),                          # custom-1 opcode
    ("f_lw", f"    li t0, {DATA + 2}\n    li t1, 0x55\n    lw t1, 0(t0)\n"),
    ("f_lh", f"    li t0, {DATA + 1}\n    lh t1, 0(t0)\n"),
    ("f_lhu", f"    li t0, {DATA + 3}\n    lhu t1, 0(t0)\n"),
    ("f_lw1", f"    li t0, {DATA}\n    lw t1, 1(t0)\n"),
    ("f_sw", f"    li t0, {DATA + 6}\n    li t1, -1\n    sw t1, 0(t0)\n"),
    ("f_sh", f"    li t0, {DATA + 9}\n    sh t1, 0(t0)\n"),
    ("f_jalr", "    li ra, 0x777\n    la t0, f_target+2\n    jalr ra, 0(t0)\n"),
    ("f_jal", "    li t1, 0x888\n    jal t1, . + 6\n"),
    ("f_beq", "    beq x0, x0, . + 10\n"),
    ("f_bnt", "    bne x0, x0, . + 6\n    li t1, 0x999\n    ecall\n"),  # not taken: no fault
    ("f_dbus_ld", "    li t0, 0x100000\n    lw t1, 0(t0)\n"),
    ("f_dbus_st", "    li t0, 0x200000\n    sw t0, -4(t0)\n"),
    ("f_ibus", "    li t0, 0x30000\n    li ra, 0x5\n    jalr ra, 0(t0)\n"),
    ("f_ibus_jal", "    li t1, 0x6\n    jal t1, 0x20000\n"),             # target fetched from EXEC
    ("f_ok", "    addi t1, t1, 1\n    ebreak\n"),
]


def fault_program() -> tuple[str, list[str]]:
    text = PROLOGUE
    for name, body in FAULT_ENTRIES:
        text += f"{name}:\n{body}    ecall\n"
    text += "f_target:\n    ecall\n"
    text += f".org {DATA}\n    .word 0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00\n"
    # Sequential execution off the end of memory: the fetch of 0x10000 is a
    # bus error after two retired instructions.
    text += f".org {MEM_SIZE - 8}\nf_ibus_seq:\n    addi t1, t1, 1\n    addi t1, t1, 1\n"
    starts = [name for name, _ in FAULT_ENTRIES] + ["f_ibus_seq"]
    return text, starts


def _enc_r(f7, rs2, rs1, f3, rd, opc):
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opc


def encoding_sweep_program() -> tuple[str, list[str]]:
    """One run per encoding: legal ones retire and reach the ECALL behind them,
    illegal ones fault (cause 1). Covers every funct3 / funct7 variant that the
    legality decode distinguishes, and every major opcode with bits [1:0] = 11
    (except JAL/JALR/BRANCH/STORE, whose all-zero forms would loop or store)."""
    enc = []
    for f3 in (1, 5):                                   # SLLI / SRLI / SRAI funct7
        for f7 in (0x00, 0x01, 0x10, 0x20, 0x21, 0x40, 0x60):
            enc.append(_enc_r(f7, 3, 0, f3, 1, 0b0010011))
    enc += [0x41F05093, 0x01F01093, 0x01F05093]         # srai/slli/srli x1, x0, 31
    for f3 in range(8):                                 # OP with funct7 != 0
        for f7 in (0x01, 0x20, 0x40, 0x7F):
            enc.append(_enc_r(f7, 2, 1, f3, 1, 0b0110011))
    enc += [(f3 << 12) | (1 << 7) | 0b0000011 for f3 in range(8)]          # loads from 0
    enc += [(f3 << 12) | 0b0100011 for f3 in range(3, 8)]                   # illegal stores
    enc += [(f3 << 12) | (8 << 8) | 0b1100011 for f3 in (2, 3)]             # branch funct3
    enc += [(f3 << 12) | 0b1100111 for f3 in range(1, 8)]                   # jalr funct3
    enc += [(f3 << 12) | 0b0001111 for f3 in range(8)]                      # fence funct3
    enc += [0x00000073, 0x00100073, 0x000000F3, 0x00008073, 0x00004073,     # SYSTEM
            0x00200073, 0x10200073, 0x10500073, 0x00101073, 0x80000073, 0x00110073]
    enc += [opc for opc in range(3, 128, 4)
            if opc not in (0b1101111, 0b1100111, 0b1100011, 0b0100011)]
    enc += [0x00000001, 0x00000002, 0x00004501]         # 16-bit (compressed) space
    text = PROLOGUE + "    ecall\n"
    starts = [0]
    for i, word in enumerate(enc):
        text += f"e{i}:\n    .word {word:#010x}\n    ecall\n"
        starts.append(f"e{i}")
    return text, starts


def edge_program(instr: str, t0: int) -> str:
    """One instruction in the last memory word, so the fetch of its successor
    (0x10000, issued while it executes) is an instruction bus error. With a
    vector command the error response waits through the vcmd_ready stall and
    the core faults after the command retires; with a load that gets a data
    bus error the error fetch is abandoned (drained after the stop, or while
    FETCH waits after the next start)."""
    return PROLOGUE + f"    li t0, {_h(t0)}\n    li t1, 0x5a\n    j edge\n" \
        f".org {MEM_SIZE - 4}\nedge:\n    {instr}\n"


# --------------------------------------------------------------------- (e)
def restart_program() -> str:
    return PROLOGUE + """
    li a0, 1
    addi a1, a1, 1
    ecall
.org 0x400
entry1:
    add a0, a0, a0
    addi a1, a1, 1
    ebreak
.org 0x8000
entry2:
    addi a1, a1, 16
    sw a1, 0x100(x0)
    ecall
"""


# --------------------------------------------------------------------- (f)
def random_program(rng, count: int) -> str:
    regs = [f"x{i}" for i in range(32) if i != 8]          # s0 is the memory base
    lines = [PROLOGUE, f"    li s0, {SCRATCH}"]
    for i in range(1, 32):
        if i != 8:
            lines.append(f"    li x{i}, {_h(rng.getrandbits(32))}")
    for i in range(0, 256, 4):
        lines.append(f"    sw x{rng.randrange(32)}, {i - 128}(s0)")
    rr = ["add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and"]
    ri = ["addi", "slti", "sltiu", "xori", "ori", "andi"]
    sh = ["slli", "srli", "srai"]
    for _ in range(count):
        kind = rng.random()
        rd = rng.choice(regs)
        r1, r2 = rng.choice(regs + ["s0"]), rng.choice(regs + ["s0"])
        if kind < 0.35:
            lines.append(f"    {rng.choice(rr)} {rd}, {r1}, {r2}")
        elif kind < 0.6:
            lines.append(f"    {rng.choice(ri)} {rd}, {r1}, {rng.randint(-2048, 2047)}")
        elif kind < 0.7:
            lines.append(f"    {rng.choice(sh)} {rd}, {r1}, {rng.randrange(32)}")
        elif kind < 0.75:
            lines.append(f"    {rng.choice(['lui', 'auipc'])} {rd}, {rng.getrandbits(20):#x}")
        elif kind < 0.88:
            op, size = rng.choice([("lw", 4), ("lh", 2), ("lhu", 2), ("lb", 1), ("lbu", 1)])
            lines.append(f"    {op} {rd}, {rng.randrange(-128, 128, size)}(s0)")
        elif kind < 0.93:
            op, size = rng.choice([("sw", 4), ("sh", 2), ("sb", 1)])
            lines.append(f"    {op} {r1}, {rng.randrange(-128, 128, size)}(s0)")
        else:
            # Forward branch over one instruction; equal operands half the time.
            op = rng.choice(["beq", "bne", "blt", "bge", "bltu", "bgeu"])
            if rng.random() < 0.3:
                r2 = r1
            lab = f"rb{len(lines)}"
            lines += [f"    {op} {r1}, {r2}, {lab}", f"    xori {rd}, {rd}, 0x5a5", f"{lab}:"]
    lines.append("    ecall")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- (g)
def cpi_programs() -> list[tuple[str, str]]:
    body = {
        "empty": "",
        "alu": "".join(f"    {op} t0, t0, t1\n" for op in ("add", "xor", "sll", "slt") * 8)
               + "".join("    addi t1, t1, 3\n    lui t2, 5\n    auipc t3, 1\n    srai t1, t1, 1\n"
                         for _ in range(8)),
        "load": f"    li s1, {DATA}\n" + "    lw t0, 0(s1)\n    lbu t1, 1(s1)\n" * 16,
        "store": f"    li s1, {DATA}\n" + "    sw t0, 0(s1)\n    sb t1, 3(s1)\n" * 16,
        "branch": "".join(f"    beq x0, x0, b{i}\nb{i}:\n    bne x0, x0, b{i}\n" for i in range(16)),
        "jump": "".join(f"    jal x0, j{i}\nj{i}:\n" for i in range(16))
                + "".join(f"    la t0, k{i}\n    jalr ra, 0(t0)\nk{i}:\n" for i in range(8)),
        "vector": "".join(f"    vadd.i8 v{i % 8}, v1, v2\n" for i in range(32)),
        "vfence": "    vfence\n" * 16,
    }
    return [(f"cpi_{k}", PROLOGUE + v + "    ecall\n") for k, v in body.items()]


def cpi_mix_program() -> str:
    """Every instruction class once per loop iteration (latency > 1 model)."""
    return PROLOGUE + f"""
    li s1, {DATA}
    li a0, 6
mix:
    lw t0, 0(s1)
    sw t0, 4(s1)
    add t1, t1, t0
    vadd.i8 v1, v2, v3
    vfence
    jal ra, mixf
    addi a0, a0, -1
    bnez a0, mix
    ecall
mixf:
    ret
"""


# --------------------------------------------------------------------- build
def build_case(name, source, starts=None, latency=-1, d_latency=None, check_cycles=True):
    d_latency = latency if d_latency is None else d_latency
    prog = assemble(source)
    image = prog.image()
    iss = RV32ISS(MEM_SIZE, latency=max(latency, 0))
    iss.load_words(image)
    runs = []
    entries = starts if starts is not None else [0]
    for entry in entries:
        pc = prog.symbols[entry] if isinstance(entry, str) else entry
        res = iss.run(pc)
        runs.append((pc, res))
    mem_words = sorted(set(image) | iss.written)
    final_mem = [(a, iss.word(a)) for a in mem_words]
    return {
        "name": name, "latency": (latency, d_latency),
        "check_cycles": latency >= 0 and latency == d_latency and check_cycles,
        "init": sorted(image.items()), "runs": runs,
        "vcmds": [c.as_tuple() for c in iss.vcmds], "mem": final_mem,
    }


def build_cases():
    rng = random.Random(SEED)
    cases = [
        build_case("coverage", coverage_program()),
        build_case("memcpy", memcpy_program(rng)),
        build_case("bubble_sort", bubble_program(rng)),
        build_case("fibonacci", fib_program()),
        build_case("vector", vector_program()),
    ]
    text, starts = fault_program()
    # First run starts at 0 (prologue, then the first fault); then a misaligned
    # start PC and a start PC outside memory (bus error on the first fetch,
    # fault_pc = start_pc).
    prog = assemble(text)
    fault_starts = [0] + starts[1:] + [prog.symbols["f_ok"] + 2, "f_ok", 0x40000]
    cases.append(build_case("faults", text, fault_starts))
    # The same runs with fixed latencies: the cycle model of every stop with a
    # 1-cycle and a same-cycle memory; and a slow instruction port with a
    # same-cycle data port, where the fetch issued by a load / store that then
    # gets a data bus error is still in flight after the stop (testbench
    # restarts immediately): it drains while stopped (3) or while FETCH waits
    # for it (6).
    cases.append(build_case("faults_ideal", text, fault_starts, latency=1))
    cases.append(build_case("faults_zl", text, fault_starts, latency=0))
    cases.append(build_case("faults_slow3", text, fault_starts, latency=3, d_latency=0))
    cases.append(build_case("faults_slow6", text, fault_starts, latency=6, d_latency=0))
    # Successor fetch at the end of memory: held through a vector-command
    # stall, or abandoned (with its error) by a data bus error.
    edge_vcmd = edge_program("vsplat.i32 v1, t1", DATA)
    cases.append(build_case("edge_vcmd", edge_vcmd, [0] * 8))
    cases.append(build_case("edge_vcmd_ideal", edge_vcmd, [0] * 2, latency=1))
    edge_dbus = edge_program("lw t1, 0(t0)", 0x100000)
    cases.append(build_case("edge_dbus", edge_dbus, [0] * 4))
    cases.append(build_case("edge_dbus_ideal", edge_dbus, [0] * 2, latency=1))
    cases.append(build_case("edge_dbus_slow3", edge_dbus, [0] * 2, latency=3, d_latency=0))
    cases.append(build_case("edge_dbus_slow6", edge_dbus, [0] * 2, latency=6, d_latency=0))
    text, starts = encoding_sweep_program()
    cases.append(build_case("encodings", text, starts))
    cases.append(build_case("restart", restart_program(), [0, "entry1", "entry2", "entry1", 0]))
    for k in range(3):
        cases.append(build_case(f"random{k}", random_program(rng, 400)))
    cases.append(build_case("random_zl", random_program(rng, 200), latency=0))
    for name, source in cpi_programs():
        cases.append(build_case(name, source, latency=1))
    for name, source in cpi_programs():
        cases.append(build_case(name.replace("cpi_", "cpi0_"), source, latency=0))
    cases.append(build_case("cpi3_mix", cpi_mix_program(), latency=3))
    return cases


def render(cases) -> str:
    out = ["# tb_npu_scalar_core vectors: generated by src/test/vectors/generate_scalar_vectors.py\n"]
    for case in cases:
        out.append(f"CASE {case['name']} {case['latency'][0]} {case['latency'][1]} "
                   f"{len(case['init'])}\n")
        out.extend(f"{a:08x} {w:08x}\n" for a, w in case["init"])
        out.append(f"RUNS {len(case['runs'])}\n")
        for pc, res in case["runs"]:
            status = 1 if res.status == "halt" else 2
            cycles = res.cycles if case["check_cycles"] else -1
            out.append(f"RUN {pc:08x} {status} {res.cause} {res.stop_pc:08x} {res.instret} {cycles}\n")
            out.extend(f"{r:08x}\n" for r in res.regs[1:])
        out.append(f"VCMDS {len(case['vcmds'])}\n")
        out.extend(f"{op} {t} {vd} {a} {b} {s:08x}\n" for op, t, vd, a, b, s in case["vcmds"])
        out.append(f"MEM {len(case['mem'])}\n")
        out.extend(f"{a:08x} {w:08x}\n" for a, w in case["mem"])
    out.append("DONE\n")
    return "".join(out)


def write_vectors(path: Path = OUTPUT):
    path.write_text(render(build_cases()), encoding="ascii")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the committed file is up to date instead of writing it")
    args = parser.parse_args(argv)
    text = render(build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="ascii") != text:
            print(f"{OUTPUT} is stale; rerun the generator", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
