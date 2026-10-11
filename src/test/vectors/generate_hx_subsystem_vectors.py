"""Generate the test script for tb_npu_hx_subsystem (issue #136).

The testbench is a small script interpreter around npu_hx_subsystem: an AXI4
memory model (64 KB of DDR at address 0; any access at or past 0x10000
answers SLVERR) and an AXI4-Lite host BFM. This generator writes the script:
programs are assembled with src/test/model/rv32_asm.py and their expected
results come from the reference ISS (src/test/model/rv32_iss.py, extended
here with the subsystem's TCM window) and the vector unit model
(src/test/model/hx_vector.py).

The testbench instantiates four subsystem configurations (one active at a
time); CONFIG selects one and resets it:
    0  all units                       CAPS 0x0010087F
    1  HX_ENABLE_VECTOR = 0            CAPS 0x0010007D
    2  HX_ENABLE_L2 = 0                CAPS 0x0010086F
    3  HX_ENABLE_DMA = 0               CAPS 0x0010083F

Script (one command per line, numbers hex unless marked dec, '#' comments)::

    CONFIG <idx dec>              select a configuration, reset it
    CASE <name>                   progress label
    CFG <ready_pct dec> <max_lat dec>
                                  memory model: each ready / response-valid
                                  is high with probability ready_pct %, each
                                  burst starts after 0..max_lat cycles
                                  (100 0 = zero-wait slave)
    RST                           reset the active configuration
    MW <addr> <data>              backdoor DDR word write
    MC <addr> <data>              backdoor DDR word check
    W <off> <data>                host write, all byte strobes
    WS <off> <data> <strb>        host write with byte strobes
    R <off> <exp> <mask>          host read, check (rdata & mask) == exp
    RW <woff> <wdata> <roff> <exp> <mask>
                                  host write and host read issued
                                  concurrently (different registers)
    HFAST <0|1>                   host BFM: 1 = back-to-back transactions,
                                  BREADY / RREADY high from the start;
                                  0 = random delays (default)
    P <off> <mask> <val> <n dec>  poll until (rdata & mask) == val, <= n reads
    PN <off> <mask> <n dec>       poll until (rdata & mask) != 0, <= n reads
    IRQ <0|1>                     check the irq output
    T0 <label>                    start a cycle stopwatch
    T1 <label>                    print the cycles since T0
    SHOW <off> <label>            host read, print the value (decimal)
    END

    python3 -m src.test.vectors.generate_hx_subsystem_vectors [--check]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from src.test.model import hx_numeric as hx
from src.test.model import hx_vector as hv
from src.test.model.rv32_asm import assemble
from src.test.model.rv32_iss import BusError, RV32ISS

OUTPUT = Path(__file__).with_name("hx_subsystem_script.txt")
SEED = 0x1360

DDR_SIZE = 0x10000
TCM_WORDS = 1024
TCM_BASE = 0x4000_0000
MMIO_BASE = 0x8000_0000

# Register offsets (architecture.md, npu_hx_regs)
MAGIC, VERSION, CAPS, CONTROL, STATUS, START_PC, CAUSE, FAULT_PC = range(0, 0x20, 4)
CYCLES, INSTRET = 0x20, 0x24
L1I_HITS, L1I_MISSES, L1D_HITS, L1D_MISSES, L2_HITS, L2_MISSES = range(0x28, 0x40, 4)
DMA_DDR, DMA_TCM, DMA_LEN, IRQ_EN = 0x40, 0x44, 0x48, 0x4C
MBOX = [0x50, 0x54, 0x58, 0x5C]
VEC_CYCLES = 0x60
TCM_WIN = 0x8000

C_START, C_FLUSH, C_DMA, C_CLEAR = 1, 2, 4, 8
S_RUNNING, S_HALTED, S_FAULT, S_VBUSY, S_DBUSY, S_FBUSY = 1, 2, 4, 8, 0x10, 0x20
S_VILL, S_DERR, S_DDONE, S_BERR, S_HEVT, S_FEVT = 0x40, 0x80, 0x100, 0x200, 0x400, 0x800

CONFIGS = [
    # (name, vector, l2, dma)
    ("full", 1, 1, 1),
    ("no_vector", 0, 1, 1),
    ("no_l2", 1, 0, 1),
    ("no_dma", 1, 1, 0),
]

POLL = 20000        # polls before the testbench gives up


def caps_of(vec: int, l2: int, dma: int) -> int:
    bits = 1 | (vec << 1) | (1 << 2) | (1 << 3) | (l2 << 4) | (1 << 5) | (dma << 6)
    return ((TCM_WORDS // 64) << 16) | ((8 if vec else 0) << 8) | bits


class HxISS(RV32ISS):
    """The reference ISS with the subsystem's data map: DDR at 0 (64 KB) and
    the TCM window at 0x4000_0000 (byte array of TCM_WORDS * 8 bytes). Any
    other data address is a bus error (the RTL answers it with d_resp_err)."""

    def __init__(self, ddr: bytearray, tcm: bytearray):
        super().__init__(mem_size=1 << 32, mem=ddr)
        self.tcm = tcm
        self.ddr_written: set[int] = set()

    def _region(self, addr: int, size: int):
        if 0 <= addr and addr + size <= DDR_SIZE:
            return self.mem, addr
        if TCM_BASE <= addr and addr + size <= TCM_BASE + len(self.tcm):
            return self.tcm, addr - TCM_BASE
        raise BusError

    def word(self, addr: int) -> int:
        buf, off = self._region(addr, 4)
        return int.from_bytes(buf[off:off + 4], "little")

    def _store(self, addr: int, value: int, size: int):
        buf, off = self._region(addr, size)
        buf[off:off + size] = (value & ((1 << (8 * size)) - 1)).to_bytes(size, "little")
        if buf is self.mem:
            self.ddr_written.add(addr & ~3)


class Script:
    def __init__(self):
        self.lines: list[str] = []

    def emit(self, text: str):
        self.lines.append(text)

    def comment(self, text: str):
        self.emit(f"# {text}")

    def config(self, idx: int):
        self.emit(f"CONFIG {idx}")

    def case(self, name: str):
        self.emit(f"CASE {name}")

    def cfg(self, pct: int, lat: int):
        self.emit(f"CFG {pct} {lat}")

    def rst(self):
        self.emit("RST")

    def mw(self, addr: int, data: int):
        self.emit(f"MW {addr:08x} {data & 0xFFFFFFFF:08x}")

    def mc(self, addr: int, data: int):
        self.emit(f"MC {addr:08x} {data & 0xFFFFFFFF:08x}")

    def w(self, off: int, data: int, strb: int = 0xF):
        if strb == 0xF:
            self.emit(f"W {off:04x} {data & 0xFFFFFFFF:08x}")
        else:
            self.emit(f"WS {off:04x} {data & 0xFFFFFFFF:08x} {strb:x}")

    def r(self, off: int, exp: int, mask: int = 0xFFFFFFFF):
        self.emit(f"R {off:04x} {exp & mask:08x} {mask:08x}")

    def rw(self, woff: int, wdata: int, roff: int, exp: int, mask: int = 0xFFFFFFFF):
        self.emit(f"RW {woff:04x} {wdata & 0xFFFFFFFF:08x} {roff:04x} {exp & mask:08x} {mask:08x}")

    def hfast(self, on: bool):
        self.emit(f"HFAST {int(on)}")

    def poll(self, off: int, mask: int, val: int, n: int = POLL):
        self.emit(f"P {off:04x} {mask:08x} {val:08x} {n}")

    def poll_nz(self, off: int, mask: int, n: int = POLL):
        self.emit(f"PN {off:04x} {mask:08x} {n}")

    def irq(self, level: int):
        self.emit(f"IRQ {level}")

    def t0(self, label: str):
        self.emit(f"T0 {label}")

    def t1(self, label: str):
        self.emit(f"T1 {label}")

    def show(self, off: int, label: str):
        self.emit(f"SHOW {off:04x} {label}")

    # ------------------------------------------------------------ helpers
    def load_image(self, image: dict[int, int]):
        for addr in sorted(image):
            self.mw(addr, image[addr])

    def load_bytes(self, base: int, data: bytes):
        for off in range(0, len(data), 4):
            self.mw(base + off, int.from_bytes(data[off:off + 4], "little"))

    def check_bytes(self, base: int, data: bytes):
        for off in range(0, len(data), 4):
            self.mc(base + off, int.from_bytes(data[off:off + 4], "little"))

    def flush(self):
        self.w(CONTROL, C_FLUSH)
        self.poll(STATUS, S_FBUSY, 0)

    def start(self, pc: int):
        self.w(START_PC, pc)
        self.w(CONTROL, C_START)

    def wait_stop(self):
        self.poll_nz(STATUS, S_HALTED | S_FAULT)

    def run(self, pc: int, halt_cause: int | None = 0, fault_cause: int | None = None,
            fault_pc: int | None = None):
        """START at pc, wait for halt/fault, check the cause."""
        self.start(pc)
        self.wait_stop()
        if fault_cause is None:
            self.r(STATUS, S_HALTED, S_RUNNING | S_HALTED | S_FAULT)
            self.r(CAUSE, (halt_cause or 0) << 8, 0x107)
        else:
            self.r(STATUS, S_FAULT, S_RUNNING | S_HALTED | S_FAULT)
            self.r(CAUSE, fault_cause, 0x107)
            if fault_pc is not None:
                self.r(FAULT_PC, fault_pc)

    def dma(self, ddr: int, tcm: int, words: int, to_ddr: bool, wait: bool = True):
        self.w(DMA_DDR, ddr)
        self.w(DMA_TCM, tcm)
        self.w(DMA_LEN, (0x8000_0000 if to_ddr else 0) | words)
        self.w(CONTROL, C_DMA)
        if wait:
            self.poll_nz(STATUS, S_DDONE)
            self.r(STATUS, 0, S_DBUSY | S_DERR)

    def show_counters(self, tag: str):
        for off, name in ((CYCLES, "cycles"), (INSTRET, "instret"),
                          (L1I_HITS, "l1i_hits"), (L1I_MISSES, "l1i_misses"),
                          (L1D_HITS, "l1d_hits"), (L1D_MISSES, "l1d_misses"),
                          (L2_HITS, "l2_hits"), (L2_MISSES, "l2_misses")):
            self.show(off, f"{tag}.{name}")

    def render(self) -> str:
        head = ("# tb_npu_hx_subsystem script: generated by "
                "src/test/vectors/generate_hx_subsystem_vectors.py\n")
        return head + "\n".join(self.lines) + "\nEND\n"


def words_to_bytes(words64: list[int]) -> bytes:
    return b"".join((w & (2**64 - 1)).to_bytes(8, "little") for w in words64)


# ---------------------------------------------------------------- programs
PC_B = 0x0000      # (b) sum / copy / sort in DDR
PC_C = 0x0800      # (c) vector kernels on the TCM
PC_D = 0x1000      # (d) scalar-driven DMA, mailbox
PC_E = 0x1800      # (e) scalar TCM window byte / half / word accesses
PC_F = 0x2000      # (f) faults: 0x40 bytes per program
PC_G = 0x2800      # (g) SLVERR loads / stores
PC_H = 0x3000      # (h) instruction cache refill
PC_H2 = 0x3400

D_B = 0x8000       # (b) data: src 0x8000, dst 0x9000 (same L1 / L2 set), sort 0x8400
D_C_IN = 0x9800    # (c) input words for the DMA
D_C_OUT = 0xA000   # (c) results
D_D_SRC, D_D_DST = 0xB000, 0xB400
D_E_OUT = 0xC000
D_H = 0xD000

PROG_B = f"""
    # copy with transform: dst[i] = (src[i] << 1) + src[i] + i, 48 words;
    # dst = src + 4 KB maps to the same L1 D$ line and L2 set (dirty evictions)
    li   s0, {D_B:#x}
    li   s1, {D_B + 0x1000:#x}
    li   t0, 0
    li   t1, 48
    li   s2, 0              # sum of src
copy:
    lw   t2, 0(s0)
    add  s2, s2, t2
    slli t3, t2, 1
    add  t3, t3, t2
    add  t3, t3, t0
    sw   t3, 0(s1)
    addi s0, s0, 4
    addi s1, s1, 4
    addi t0, t0, 1
    bne  t0, t1, copy
    li   s0, {D_B + 0x1400:#x}
    sw   s2, 0(s0)          # sum after the copy
    # bubble sort (signed) of 12 words at D_B + 0x400
    li   a0, {D_B + 0x400:#x}
    li   a1, 11
outer:
    mv   a2, a0
    mv   a3, a1
inner:
    lw   t2, 0(a2)
    lw   t3, 4(a2)
    ble  t2, t3, noswap
    sw   t3, 0(a2)
    sw   t2, 4(a2)
noswap:
    addi a2, a2, 4
    addi a3, a3, -1
    bnez a3, inner
    addi a1, a1, -1
    bnez a1, outer
    # sub-word stores: bytes and halves of the sum into D_B + 0x1408..
    li   s0, {D_B + 0x1408:#x}
    sb   s2, 0(s0)
    srli t2, s2, 8
    sb   t2, 5(s0)
    sh   s2, 10(s0)
    lbu  t3, 5(s0)
    lh   t4, 10(s0)
    sw   t3, 12(s0)
    sw   t4, 16(s0)
    ecall
"""


def prog_c_source(a_fp16: int) -> str:
    return f"""
    # INT8 residual add with saturation: y = sat(x + r), 32 words
    li   a0, 0
    li   a1, 32
    li   a2, 64
    li   t0, 32
loop1:
    vld      v1, a0
    vld      v2, a1
    vadd.i8  v3, v1, v2
    vst      v3, a2
    addi a0, a0, 1
    addi a1, a1, 1
    addi a2, a2, 1
    addi t0, t0, -1
    bnez t0, loop1
    # FP16 a*x + y, 32 words (two roundings: vmul then vadd)
    li   t1, {a_fp16:#x}
    vsplat.f16 v4, t1
    li   a0, 96
    li   a1, 128
    li   a2, 160
    li   t0, 32
loop2:
    vld      v1, a0
    vld      v2, a1
    vmul.f16 v5, v4, v1
    vadd.f16 v6, v5, v2
    vst      v6, a2
    addi a0, a0, 1
    addi a1, a1, 1
    addi a2, a2, 1
    addi t0, t0, -1
    bnez t0, loop2
    vfence
    ecall
"""


PROG_D = f"""
    # mailbox: MBOX0 = src DDR, MBOX1 = words, MBOX2 = dst DDR
    lui  s0, 0x80000
    lw   a0, 0x50(s0)
    lw   a1, 0x54(s0)
    lw   a2, 0x58(s0)
    sw   a0, 0x40(s0)       # DMA_DDR_ADDR
    li   t0, 8
    sw   t0, 0x44(s0)       # DMA_TCM_ADDR = word 8
    sw   a1, 0x48(s0)       # DMA_LEN, DDR -> TCM
    li   t0, 4
    sw   t0, 0x0c(s0)       # CONTROL.DMA_START
wait1:
    lw   t1, 0x10(s0)
    andi t1, t1, 0x100      # STATUS.dma_done
    beqz t1, wait1
    # add 1 to every 32-bit element through the TCM window, sum the inputs
    li   s1, {TCM_BASE + 8 * 8:#x}
    slli t2, a1, 1
    li   t3, 0
loop:
    lw   t4, 0(s1)
    add  t3, t3, t4
    addi t4, t4, 1
    sw   t4, 0(s1)
    addi s1, s1, 4
    addi t2, t2, -1
    bnez t2, loop
    # DMA back: TCM word 8 -> dst
    sw   a2, 0x40(s0)
    lui  t0, 0x80000
    or   t0, t0, a1
    sw   t0, 0x48(s0)       # DMA_LEN, TCM -> DDR
    li   t0, 4
    sw   t0, 0x0c(s0)
wait2:
    lw   t1, 0x10(s0)
    andi t1, t1, 0x100
    beqz t1, wait2
    lw   t5, 0(s0)          # MAGIC through MMIO
    sw   t5, 0x58(s0)       # MBOX2 = MAGIC
    sw   t3, 0x5c(s0)       # MBOX3 = sum
    li   t0, 0x600d
    sh   t0, 0x50(s0)       # MBOX0[15:0] = 0x600d (half store to MMIO)
    ecall
"""

PROG_E = f"""
    # sub-word accesses to the TCM window, results to DDR
    li   s0, {TCM_BASE:#x}
    li   s1, {D_E_OUT:#x}
    lw   t0, 0(s0)
    lw   t1, 4(s0)
    lb   t2, 9(s0)
    lbu  t3, 14(s0)
    lh   t4, 18(s0)
    lhu  t5, 22(s0)
    sw   t0, 0(s1)
    sw   t1, 4(s1)
    sw   t2, 8(s1)
    sw   t3, 12(s1)
    sw   t4, 16(s1)
    sw   t5, 20(s1)
    li   t0, 0xa5
    sb   t0, 32(s0)
    li   t0, 0x5a
    sb   t0, 37(s0)
    li   t0, 0x1234
    sh   t0, 42(s0)
    li   t0, 0xbeef
    sh   t0, 44(s0)
    li   t0, -2
    sw   t0, 52(s0)
    lw   t1, 32(s0)
    lw   t2, 36(s0)
    sw   t1, 24(s1)
    sw   t2, 28(s1)
    li   t0, {TCM_BASE + 8 * (TCM_WORDS - 1) + 4:#x}
    li   t1, 0x7e57
    sw   t1, 0(t0)          # last TCM word, upper lane
    lw   t2, 0(t0)
    sw   t2, 32(s1)
    ecall
"""


PC_E2 = 0x1C00     # (e) scalar TCM traffic while the DMA streams
PC_I = 0x3800      # (i) scalar MMIO traffic while the host accesses registers
D_E2_DMA_OUT = 0x6000

PROG_E2 = f"""
    # TCM window traffic during DMA bursts on port B: 64 times
    # dst[i] = src[i % 32] ^ i (32-bit lanes; src = TCM bytes 0..127,
    # dst = TCM word 128..), checksum to DDR
    li   s0, {TCM_BASE:#x}
    li   s1, {TCM_BASE + 8 * 128:#x}
    li   t0, 64
    li   t1, 0
    li   t3, 0
loop:
    andi t2, t1, 0x1f
    slli t2, t2, 2
    add  t4, s0, t2
    lw   t5, 0(t4)
    xor  t5, t5, t1
    add  t3, t3, t5
    slli t6, t1, 2
    add  t6, s1, t6
    sw   t5, 0(t6)
    addi t1, t1, 1
    bne  t1, t0, loop
    li   s2, {D_E_OUT + 0x40:#x}
    sw   t3, 0(s2)
    ecall
"""

PROG_I = """
    # MMIO traffic: MBOX0 iterations of MBOX3 += 1 and a STATUS / MAGIC read;
    # MBOX0 = number of reads that saw STATUS.running, MBOX1 is the host's
    lui  s0, 0x80000
    lw   t0, 0x50(s0)
    li   t1, 0
    li   s1, 0x314e5848
loop:
    lw   t2, 0x5c(s0)
    addi t2, t2, 1
    sw   t2, 0x5c(s0)
    lw   t3, 0x10(s0)
    andi t3, t3, 1
    add  t1, t1, t3
    lw   t4, 0(s0)
    bne  t4, s1, bad
    addi t0, t0, -1
    bnez t0, loop
    sw   t1, 0x50(s0)
    ecall
bad:
    ebreak
"""


def fault_programs():
    """(name, source, fault_cause or None, halt_cause, fault offset label)."""
    return [
        ("misaligned_load", """
    li   t0, 0x8002
    nop
fault:
    lw   t1, 0(t0)
    ecall
""", 3, 0),
        ("ebreak", """
    li   t0, 1
fault:
    ebreak
""", None, 1),
        ("ecall", """
    li   t0, 2
fault:
    ecall
""", None, 0),
        ("illegal", """
    li   t0, 3
fault:
    .word 0xffffffff
    ecall
""", 1, 0),
        ("misaligned_store", """
    li   t0, 0x8001
fault:
    sh   t0, 0(t0)
    ecall
""", 4, 0),
        ("misaligned_jump", """
    li   t0, 0x2002
fault:
    jalr x0, 0(t0)
    ecall
""", 2, 0),
        ("unmapped_load", """
    lui  t0, 0xc0000
fault:
    lw   t1, 0(t0)
    ecall
""", 6, 0),
        ("tcm_past_end", f"""
    li   t0, {TCM_BASE + 8 * TCM_WORDS:#x}
fault:
    sw   t0, 0(t0)
    ecall
""", 6, 0),
        ("mmio_past_end", """
    lui  t0, 0x80001
fault:
    lw   t1, 0(t0)
    ecall
""", 6, 0),
    ]


PROG_G = """
    # loads and a store past the end of the model's DDR (SLVERR)
    li   t0, 0x10000
    lw   t1, 0(t0)
    li   t2, 0x77
    sw   t2, 0x40(t0)
    li   t3, 0xd000
    sw   t2, 0(t3)
    ecall
"""


def prog_h(value: int) -> str:
    return f"""
    li   t0, {value:#x}
    li   t1, {D_H:#x}
    sw   t0, 0(t1)
    ecall
"""


def prog_h2(value: int) -> str:
    return f"""
    li   t0, {value:#x}
    li   t1, {D_H:#x}
    sw   t0, 4(t1)
    lw   t2, 0(t1)
    sw   t2, 8(t1)
    ecall
"""


# ---------------------------------------------------------------- cases
def case_regs(s: Script, cfg: int):
    name, vec, l2, dma = CONFIGS[cfg]
    s.case(f"regs_{name}")
    for off, exp in ((MAGIC, 0x314E5848), (VERSION, 0x00010000), (CAPS, caps_of(vec, l2, dma))):
        s.r(off, exp)
    if cfg != 0:
        return
    # reset values
    for off in [CONTROL, STATUS, START_PC, CAUSE, FAULT_PC, CYCLES, INSTRET,
                L1I_HITS, L1I_MISSES, L1D_HITS, L1D_MISSES, L2_HITS, L2_MISSES,
                DMA_DDR, DMA_TCM, DMA_LEN, IRQ_EN] + MBOX + [VEC_CYCLES, 0x64, 0x7C,
                                                         0x80, 0xFFC, 0x1000, 0x7FFC]:
        s.r(off, 0)
    s.irq(0)
    # read/write registers and their widths
    rw = [(START_PC, 0x00001234, 0xFFFFFFFF), (DMA_DDR, 0xFFFFFFFF, 0xFFFFFFFF),
          (DMA_TCM, 0x12345678, 0xFFFFFFFF), (DMA_LEN, 0xFFFFFFFF, 0x8000FFFF),
          (IRQ_EN, 0xFFFFFFFF, 0x7), (MBOX[0], 0xDEADBEEF, 0xFFFFFFFF),
          (MBOX[1], 0x01234567, 0xFFFFFFFF), (MBOX[2], 0x89ABCDEF, 0xFFFFFFFF),
          (MBOX[3], 0xA5A55A5A, 0xFFFFFFFF)]
    for off, val, mask in rw:
        s.w(off, val)
    for off, val, mask in rw:
        s.r(off, val & mask)
    # byte strobes
    s.w(MBOX[0], 0x11223344, 0x5)       # bytes 0 and 2
    s.r(MBOX[0], 0xDE22BE44)
    s.w(DMA_LEN, 0x00000000, 0x8)       # dir only
    s.r(DMA_LEN, 0x0000FFFF)
    # writes to read-only registers are ignored
    for off in (MAGIC, VERSION, CAPS, STATUS, CAUSE, CYCLES, VEC_CYCLES):
        s.w(off, 0x5555AAAA)
    s.r(MAGIC, 0x314E5848)
    s.r(CAPS, caps_of(vec, l2, dma))
    s.r(STATUS, 0)
    s.r(CYCLES, 0)
    s.irq(0)                             # IRQ_ENABLE = 7 but no event
    s.w(IRQ_EN, 0)
    s.w(IRQ_EN, 0x7, 0xE)                # IRQ_ENABLE needs byte 0
    s.r(IRQ_EN, 0)
    # unmapped register offsets: writes are ignored (no aliasing of offset
    # bits above the 32-register block onto the registers)
    for off in (0x064, 0x07C, 0x080, 0x0FD4, 0x1050, 0x2014, 0x7FD0):
        s.w(off, 0x5555AAAA)
    expect = {off: val & mask for off, val, mask in rw}
    expect.update({MBOX[0]: 0xDE22BE44, DMA_LEN: 0x0000FFFF, IRQ_EN: 0})
    for off in sorted(expect):
        s.r(off, expect[off])
    # random register traffic on the mailboxes (read-after-write)
    rng = random.Random(SEED + 1)
    shadow = {off: 0 for off in MBOX}
    shadow[MBOX[0]] = 0xDE22BE44
    shadow[MBOX[1]] = 0x01234567
    shadow[MBOX[2]] = 0x89ABCDEF
    shadow[MBOX[3]] = 0xA5A55A5A
    for _ in range(64):
        off = rng.choice(MBOX)
        if rng.random() < 0.5:
            val = rng.getrandbits(32)
            strb = rng.choice([0xF, 0xF, 0x1, 0x2, 0x4, 0x8, 0x3, 0xC, 0x0])
            s.w(off, val, strb)
            cur = shadow[off]
            for b in range(4):
                if strb >> b & 1:
                    cur = (cur & ~(0xFF << 8 * b)) | (val & (0xFF << 8 * b))
            shadow[off] = cur
        else:
            s.r(off, shadow[off])
    # back-to-back and concurrent host traffic (a posted write and the next
    # read may overlap on the Cortex-A9's GP0 port)
    host_traffic(s, rng, shadow, 96, fast=True)
    host_traffic(s, rng, shadow, 32, fast=False)
    # reset clears everything again
    s.rst()
    for off in [START_PC, DMA_DDR, DMA_TCM, DMA_LEN, IRQ_EN] + MBOX:
        s.r(off, 0)


def host_traffic(s: Script, rng: random.Random, shadow: dict, n: int, fast: bool,
                 window: int | None = None):
    """n random host accesses on the registers in ``shadow`` (offset ->
    value): writes, reads, and concurrent write + read pairs (RW) of two
    different registers; with ``window`` (a TCM byte offset) the pairs also
    mix a register access with a TCM window access. Updates ``shadow``."""
    regs = sorted(shadow)
    s.hfast(fast)
    for _ in range(n):
        kind = rng.random()
        if kind < 0.3:
            off = rng.choice(regs)
            val = rng.getrandbits(32)
            s.w(off, val)
            shadow[off] = val
        elif kind < 0.55:
            off = rng.choice(regs)
            s.r(off, shadow[off])
        else:
            woff, roff = rng.sample(regs, 2)
            val = rng.getrandbits(32)
            s.rw(woff, val, roff, shadow[roff])
            shadow[woff] = val
    if window is not None:
        # register write || window read, window write || register read
        for i in range(8):
            woff = window + 4 * i
            val = rng.getrandbits(32)
            off = rng.choice(regs)
            s.rw(woff, val, off, shadow[off])
            reg_val = rng.getrandbits(32)
            s.rw(off, reg_val, woff, val)
            shadow[off] = reg_val
    s.hfast(False)


def case_b(s: Script, cfg: int, ideal: bool, tag: str):
    rng = random.Random(SEED + 2)
    prog = assemble(PROG_B, base=PC_B)
    ddr = bytearray(DDR_SIZE)
    for a, wd in prog.image().items():
        ddr[a:a + 4] = wd.to_bytes(4, "little")
    src = [rng.getrandbits(32) for _ in range(48)]
    sort_in = [rng.randint(-1000, 1000) & 0xFFFFFFFF for _ in range(12)]
    init = {}
    for i, v in enumerate(src):
        init[D_B + 4 * i] = v
    for i, v in enumerate(sort_in):
        init[D_B + 0x400 + 4 * i] = v
    for a, v in init.items():
        ddr[a:a + 4] = v.to_bytes(4, "little")
    iss = HxISS(ddr, bytearray(TCM_WORDS * 8))
    res = iss.run(PC_B)
    assert res.status == "halt" and res.cause == 0, res

    s.case(f"b_ddr_program_{tag}")
    if ideal:
        s.cfg(100, 0)
    else:
        s.cfg(60, 6)
    s.load_image(prog.image())
    for a in sorted(init):
        s.mw(a, init[a])
    for i in range(8):
        s.mw(D_B + 0x1400 + 4 * i, 0)
    s.flush()
    s.t0(f"b_{tag}")
    s.run(PC_B)
    s.t1(f"b_{tag}")
    s.r(INSTRET, res.instret)
    s.show_counters(f"b_{tag}")
    s.flush()
    for a in sorted(set(init) | iss.ddr_written):
        s.mc(a, iss.word(a))


def case_c(s: Script, cfg: int):
    rng = random.Random(SEED + 3)
    a_fp16 = 0xBA00          # -0.75
    prog = assemble(prog_c_source(a_fp16), base=PC_C)
    iss = HxISS(bytearray(DDR_SIZE), bytearray(TCM_WORDS * 8))
    iss.load_words(prog.image())
    res = iss.run(PC_C)
    assert res.status == "halt", res

    def int8_word():
        lanes = []
        for _ in range(8):
            if rng.random() < 0.3:
                lanes.append(rng.choice([127, 126, -128, -127, 100, -100]))
            else:
                lanes.append(rng.randint(-128, 127))
        return hv.join_lanes([x & 0xFF for x in lanes], 8)

    def fp16_lane():
        r = rng.random()
        if r < 0.05:
            return rng.choice([0x7C00, 0xFC00, 0x7E00, 0x0000, 0x8000, 0x0001, 0x83FF, 0x7BFF])
        sign = rng.getrandbits(1) << 15
        exp = rng.randint(8, 22)
        return sign | (exp << 10) | rng.getrandbits(10)

    def fp16_word():
        return hv.join_lanes([fp16_lane() for _ in range(4)], 16)

    # TCM words 0..159: x (0..31), r (32..63), 64..95 output, X (96..127), Y (128..159)
    tcm_in = [int8_word() for _ in range(64)] + [0] * 32 + [fp16_word() for _ in range(64)]
    model = hv.VectorUnitModel(vlen=64, nregs=8, tcm_words=TCM_WORDS)
    model.tcm[:len(tcm_in)] = list(tcm_in)
    for c in iss.vcmds:
        model.execute(hv.VectorCommand(op=c.op, typ=c.vtype, vd=c.vd, vs1=c.vs1, vs2=c.vs2,
                                       addr=c.scalar % TCM_WORDS, scalar=c.scalar))
    assert not model.illegal
    out = model.tcm[64:192]

    s.case("c_vector_dma")
    s.cfg(70, 4)
    s.load_image(prog.image())
    s.load_bytes(D_C_IN, words_to_bytes(tcm_in))
    for i in range(128 * 2):
        s.mw(D_C_OUT + 4 * i, 0xCCCCCCCC)
    s.flush()
    s.t0("c_total")
    s.t0("c_dma_in")
    s.dma(D_C_IN, 0, len(tcm_in), to_ddr=False)
    s.t1("c_dma_in")
    s.t0("c_vector_program")
    s.run(PC_C)
    s.t1("c_vector_program")
    s.r(STATUS, 0, S_VBUSY | S_VILL)
    s.r(INSTRET, res.instret)
    s.show(CYCLES, "c.cycles")
    s.show(VEC_CYCLES, "c.vec_cycles")
    s.poll_nz(VEC_CYCLES, 0xFFFFFFFF, 0)
    s.t0("c_dma_out")
    s.dma(D_C_OUT, 64, 128, to_ddr=True)
    s.t1("c_dma_out")
    s.t1("c_total")
    s.check_bytes(D_C_OUT, words_to_bytes(out))
    # spot check through the host TCM window
    for wi in (64, 95, 160, 191):
        s.r(TCM_WIN + 8 * wi, model.tcm[wi] & 0xFFFFFFFF)
        s.r(TCM_WIN + 8 * wi + 4, model.tcm[wi] >> 32)


def case_d(s: Script, cfg: int):
    rng = random.Random(SEED + 4)
    words = 16
    prog = assemble(PROG_D, base=PC_D)
    src = [rng.getrandbits(32) for _ in range(2 * words)]
    total = sum(src) & 0xFFFFFFFF
    s.case("d_scalar_dma_mailbox")
    s.cfg(80, 3)
    s.load_image(prog.image())
    for i, v in enumerate(src):
        s.mw(D_D_SRC + 4 * i, v)
        s.mw(D_D_DST + 4 * i, 0)
    s.flush()
    s.w(MBOX[0], D_D_SRC)
    s.w(MBOX[1], words)
    s.w(MBOX[2], D_D_DST)
    s.w(MBOX[3], 0)
    s.start(PC_D)
    s.poll(MBOX[0], 0xFFFF, 0x600D)
    s.wait_stop()
    s.r(STATUS, S_HALTED | S_DDONE, S_RUNNING | S_HALTED | S_FAULT | S_DBUSY | S_DERR | S_DDONE)
    s.r(MBOX[0], (D_D_SRC & 0xFFFF0000) | 0x600D)
    s.r(MBOX[2], 0x314E5848)
    s.r(MBOX[3], total)
    s.r(DMA_DDR, D_D_DST)
    s.r(DMA_TCM, 8)
    s.r(DMA_LEN, 0x8000_0000 | words)
    s.r(VEC_CYCLES, 0)      # START cleared it, no vector command since
    for i, v in enumerate(src):
        s.mc(D_D_DST + 4 * i, v + 1)
        s.r(TCM_WIN + 64 + 4 * i, v + 1)


def case_e(s: Script, cfg: int):
    rng = random.Random(SEED + 5)
    has_dma = CONFIGS[cfg][3]
    s.case("e_tcm_window")
    s.cfg(70, 3)
    tcm = bytearray(TCM_WORDS * 8)
    # host writes 16 64-bit words (32 window words), reads back
    for i in range(32):
        v = rng.getrandbits(32)
        s.w(TCM_WIN + 4 * i, v)
        tcm[4 * i:4 * i + 4] = v.to_bytes(4, "little")
    order = list(range(32))
    rng.shuffle(order)
    for i in order:
        s.r(TCM_WIN + 4 * i, int.from_bytes(tcm[4 * i:4 * i + 4], "little"))
    # byte-strobe writes
    for i, strb in ((3, 0x1), (4, 0x6), (5, 0x8), (6, 0x0)):
        v = rng.getrandbits(32)
        s.w(TCM_WIN + 4 * i, v, strb)
        for b in range(4):
            if strb >> b & 1:
                tcm[4 * i + b] = (v >> 8 * b) & 0xFF
        s.r(TCM_WIN + 4 * i, int.from_bytes(tcm[4 * i:4 * i + 4], "little"))
    # past the end of the TCM: reads 0, writes ignored (no aliasing onto word 0)
    s.w(TCM_WIN + 8 * TCM_WORDS, 0xFFFFFFFF)
    s.r(TCM_WIN + 8 * TCM_WORDS, 0)
    s.r(0xFFFC, 0)
    s.r(TCM_WIN, int.from_bytes(tcm[0:4], "little"))
    # last word, both lanes
    last = 8 * (TCM_WORDS - 1)
    s.w(TCM_WIN + last, 0x0BADF00D)
    s.w(TCM_WIN + last + 4, 0x12345678)
    s.r(TCM_WIN + last, 0x0BADF00D)
    s.r(TCM_WIN + last + 4, 0x12345678)
    tcm[last:last + 4] = (0x0BADF00D).to_bytes(4, "little")
    tcm[last + 4:last + 8] = (0x12345678).to_bytes(4, "little")

    # scalar sub-word accesses to the window (expected from the ISS)
    prog = assemble(PROG_E, base=PC_E)
    ddr = bytearray(DDR_SIZE)
    for a, wd in prog.image().items():
        ddr[a:a + 4] = wd.to_bytes(4, "little")
    iss = HxISS(ddr, tcm)
    res = iss.run(PC_E)
    assert res.status == "halt", res
    s.load_image(prog.image())
    for i in range(16):
        s.mw(D_E_OUT + 4 * i, 0)
    s.flush()
    s.run(PC_E)
    s.r(INSTRET, res.instret)
    s.flush()
    for a in sorted(iss.ddr_written):
        s.mc(a, iss.word(a))
    for i in range(16):
        s.r(TCM_WIN + 4 * i, int.from_bytes(tcm[4 * i:4 * i + 4], "little"))
    s.r(TCM_WIN + last + 4, 0x7E57)

    if not has_dma:
        return
    # host window traffic while a DMA streams into the TCM (port-B priority)
    dma_words = 256
    data = [rng.getrandbits(64) for _ in range(dma_words)]
    s.load_bytes(0x4000, words_to_bytes(data))
    s.dma(0x4000, 512, dma_words, to_ddr=False, wait=False)
    for i in range(24):
        v = rng.getrandbits(32)
        s.w(TCM_WIN + 4 * (64 + i), v)
        s.r(TCM_WIN + 4 * (64 + i), v)
    s.poll_nz(STATUS, S_DDONE)
    s.r(STATUS, 0, S_DBUSY | S_DERR)
    for wi in sorted(rng.sample(range(dma_words), 24)) + [0, dma_words - 1]:
        s.r(TCM_WIN + 8 * (512 + wi), data[wi] & 0xFFFFFFFF)
        s.r(TCM_WIN + 8 * (512 + wi) + 4, data[wi] >> 32)

    # all three port-B clients at once: the scalar core's TCM window
    # traffic (PROG_E2) runs while the DMA streams DDR -> TCM and then
    # TCM -> DDR, and the host hammers the window (back-to-back)
    prog2 = assemble(PROG_E2, base=PC_E2)
    ddr2 = bytearray(DDR_SIZE)
    for a, wd in prog2.image().items():
        ddr2[a:a + 4] = wd.to_bytes(4, "little")
    iss2 = HxISS(ddr2, tcm)
    res2 = iss2.run(PC_E2)
    assert res2.status == "halt" and res2.cause == 0, res2
    data2 = [rng.getrandbits(64) for _ in range(dma_words)]
    s.load_image(prog2.image())
    s.mw(D_E_OUT + 0x40, 0)
    s.load_bytes(0x4800, words_to_bytes(data2))
    for i in range(2 * dma_words):
        s.mw(D_E2_DMA_OUT + 4 * i, 0xCCCCCCCC)
    s.flush()
    s.dma(0x4800, 512, dma_words, to_ddr=False, wait=False)
    s.start(PC_E2)
    s.hfast(True)
    for i in range(16):
        v = rng.getrandbits(32)
        s.w(TCM_WIN + 4 * (64 + i), v)
        s.r(TCM_WIN + 4 * (64 + i), v)
    s.hfast(False)
    s.poll_nz(STATUS, S_DDONE)
    s.r(STATUS, S_RUNNING, S_RUNNING | S_DBUSY | S_DERR)   # still overlapping
    s.dma(D_E2_DMA_OUT, 512, dma_words, to_ddr=True, wait=False)
    s.hfast(True)
    for i in range(16, 24):
        v = rng.getrandbits(32)
        s.w(TCM_WIN + 4 * (64 + i), v)
        s.r(TCM_WIN + 4 * (64 + i), v)
    s.hfast(False)
    s.poll_nz(STATUS, S_DDONE)
    s.r(STATUS, 0, S_DBUSY | S_DERR)
    s.wait_stop()
    s.r(STATUS, S_HALTED, S_RUNNING | S_HALTED | S_FAULT)
    s.r(INSTRET, res2.instret)
    s.flush()
    s.mc(D_E_OUT + 0x40, iss2.word(D_E_OUT + 0x40))
    s.check_bytes(D_E2_DMA_OUT, words_to_bytes(data2))
    for wi in range(128, 160):
        s.r(TCM_WIN + 8 * wi, int.from_bytes(tcm[8 * wi:8 * wi + 4], "little"))
        s.r(TCM_WIN + 8 * wi + 4, int.from_bytes(tcm[8 * wi + 4:8 * wi + 8], "little"))


def case_f(s: Script, cfg: int):
    name, vec, l2, dma = CONFIGS[cfg]
    s.case(f"f_faults_irq_{name}")
    s.cfg(80, 2)
    progs = fault_programs()
    images = []
    for i, (pname, src, fcause, hcause) in enumerate(progs):
        base = PC_F + 0x40 * i
        prog = assemble(src, base=base)
        assert len(prog.words) * 4 <= 0x40
        images.append((pname, base, prog, fcause, hcause))
        s.load_image(prog.image())
    s.flush()
    s.irq(0)
    for pname, base, prog, fcause, hcause in images:
        fpc = prog.symbols["fault"]
        s.comment(pname)
        if fcause is None:
            s.run(base, halt_cause=hcause)
            s.r(FAULT_PC, fpc)
            s.r(STATUS, S_HEVT, S_HEVT | S_FEVT)
        else:
            s.run(base, fault_cause=fcause, fault_pc=fpc)
            s.r(STATUS, S_FEVT, S_HEVT | S_FEVT)
    # IRQ: fault event with IRQ_ENABLE.fault, CLEAR acknowledges
    _, base, prog, _, _ = images[0]
    s.w(IRQ_EN, 0x2)
    s.irq(1)                    # the last program faulted: the event is pending
    s.w(CONTROL, C_CLEAR, 0xE)  # CONTROL acts only with byte 0 written
    s.irq(1)
    s.r(STATUS, S_FEVT, S_FEVT)
    s.w(CONTROL, C_CLEAR)
    s.irq(0)
    s.r(STATUS, S_FAULT, S_FAULT | S_FEVT)   # fault itself is core state
    s.run(base, fault_cause=3)
    s.irq(1)
    s.w(IRQ_EN, 0x1)            # halt only: masked
    s.irq(0)
    s.w(CONTROL, C_CLEAR)
    # halt with IRQ_ENABLE.halt
    _, base, prog, _, _ = images[1]
    s.run(base, halt_cause=1)
    s.irq(1)
    s.w(CONTROL, C_CLEAR)
    s.irq(0)
    s.r(STATUS, S_HALTED, S_HALTED | S_HEVT)
    # an event pending while disabled raises irq once enabled
    s.w(IRQ_EN, 0x0)
    _, base, prog, _, _ = images[2]
    s.run(base, halt_cause=0)
    s.irq(0)
    s.w(IRQ_EN, 0x1)
    s.irq(1)
    s.w(CONTROL, C_CLEAR)
    s.irq(0)
    s.w(IRQ_EN, 0x0)
    # vector illegal command: sticky, cleared by CLEAR (without the vector
    # unit every command is flagged)
    vprog = assemble("""
    vcmd 20, 0, v1, v2, v3, x0
    vfence
    ecall
""" if vec else """
    li   t0, 5
    vsplat.i8 v1, t0
    ecall
""", base=PC_F + 0x3C0)
    s.load_image(vprog.image())
    s.flush()
    s.r(STATUS, 0, S_VILL)
    s.run(PC_F + 0x3C0)
    s.r(STATUS, S_VILL, S_VILL | S_VBUSY)
    s.w(CONTROL, C_CLEAR)
    s.r(STATUS, 0, S_VILL)
    if dma:
        # DMA done IRQ, then rejected descriptors
        s.w(IRQ_EN, 0x4)
        s.dma(0x8000, 0, 4, to_ddr=False)
        s.irq(1)
        s.w(CONTROL, C_CLEAR)
        s.irq(0)
        s.r(STATUS, 0, S_DDONE)
        for ddr, tcm, words in ((0x8004, 0, 4),            # misaligned DDR address
                                (0x8000, 0x10000, 1),      # TCM address bits above TCM_AW
                                (0x8000, TCM_WORDS - 2, 4),  # past the TCM end
                                (0x10000, 0, 2)):          # SLVERR from the memory model
            s.dma(ddr, tcm, words, to_ddr=False, wait=False)
            s.poll_nz(STATUS, S_DDONE)
            s.r(STATUS, S_DERR, S_DERR | S_DBUSY | S_BERR)
            s.irq(1)
            s.w(CONTROL, C_CLEAR)
        # a good descriptor clears dma_error
        s.dma(0x8000, 0, 2, to_ddr=False)
        s.irq(1)
        s.w(CONTROL, C_CLEAR)
        s.w(IRQ_EN, 0)
        s.irq(0)
    else:
        # DMA registers are absent, DMA_START is ignored
        s.w(DMA_DDR, 0x8000)
        s.w(DMA_LEN, 4)
        s.r(DMA_DDR, 0)
        s.r(DMA_LEN, 0)
        s.w(CONTROL, C_DMA)
        s.r(STATUS, 0, S_DBUSY | S_DDONE | S_DERR)


def case_g(s: Script, cfg: int):
    s.case("g_bus_slverr")
    s.cfg(80, 2)
    prog = assemble(PROG_G, base=PC_G)
    s.load_image(prog.image())
    s.mw(D_H, 0)
    s.flush()
    s.r(STATUS, 0, S_BERR)
    s.run(PC_G)                      # the core completes: no data error path
    s.r(STATUS, S_BERR, S_BERR)      # the read burst returned SLVERR
    s.w(CONTROL, C_CLEAR)
    s.r(STATUS, 0, S_BERR)
    s.flush()                        # writes the dirty line past the end back
    s.r(STATUS, S_BERR, S_BERR)      # BRESP = SLVERR
    s.mc(D_H, 0x77)
    s.w(CONTROL, C_CLEAR)
    s.r(STATUS, 0, S_BERR)


def case_h(s: Script, cfg: int):
    name = CONFIGS[cfg][0]
    s.case(f"h_icache_refill_{name}")
    s.cfg(80, 2)
    p1 = assemble(prog_h(0x111), base=PC_H)
    p2 = assemble(prog_h(0x222), base=PC_H)
    p3 = assemble(prog_h2(0x333), base=PC_H2)
    assert len(p1.words) == len(p2.words)
    s.load_image(p1.image())
    for i in range(4):
        s.mw(D_H + 4 * i, 0)
    s.flush()
    s.run(PC_H)
    s.show(L1I_MISSES, f"h_{name}.first.l1i_misses")
    # overwrite the program in DDR: without a flush the caches still hold the
    # old code (no coherence with the host)
    s.load_image(p2.image())
    s.run(PC_H)
    s.flush()
    s.mc(D_H, 0x111)
    s.run(PC_H)
    s.flush()
    s.mc(D_H, 0x222)
    # a new program at a new START_PC
    s.load_image(p3.image())
    s.flush()
    s.run(PC_H2)
    s.show_counters(f"h_{name}")
    s.flush()
    s.mc(D_H, 0x222)
    s.mc(D_H + 4, 0x333)
    s.mc(D_H + 8, 0x222)


def case_i(s: Script, cfg: int):
    """Scalar MMIO accesses collide with host register accesses on the one
    register port of npu_hx_regs (the scalar access has precedence)."""
    name = CONFIGS[cfg][0]
    rng = random.Random(SEED + 6)
    iters = 60
    prog = assemble(PROG_I, base=PC_I)
    s.case(f"i_mmio_host_concurrency_{name}")
    s.cfg(90, 1)
    s.load_image(prog.image())
    s.flush()
    s.w(MBOX[0], iters)
    s.w(MBOX[3], 0)
    shadow = {MBOX[1]: 0, MBOX[2]: 0, START_PC: PC_I}
    if CONFIGS[cfg][3]:
        shadow.update({DMA_DDR: 0, DMA_TCM: 0})
    for off, val in shadow.items():
        s.w(off, val)
    s.w(CONTROL, C_START)
    host_traffic(s, rng, shadow, 160, fast=True, window=TCM_WIN + 8 * 300)
    s.r(STATUS, S_RUNNING, S_RUNNING)    # the traffic overlapped the program
    s.wait_stop()
    s.r(STATUS, S_HALTED, S_RUNNING | S_HALTED | S_FAULT)
    s.r(CAUSE, 0, 0x107)
    s.r(MBOX[0], iters)
    s.r(MBOX[3], iters)
    for off in sorted(shadow):
        s.r(off, shadow[off])


def build_script() -> Script:
    s = Script()
    # configuration 0: every unit, every scenario
    s.config(0)
    case_regs(s, 0)
    case_b(s, 0, ideal=True, tag="full_ideal")
    case_b(s, 0, ideal=False, tag="full_random")
    case_c(s, 0)
    case_d(s, 0)
    case_e(s, 0)
    case_f(s, 0)
    case_g(s, 0)
    case_h(s, 0)
    case_i(s, 0)
    # configuration 1: no vector unit
    s.config(1)
    case_regs(s, 1)
    case_f(s, 1)
    # configuration 2: no L2 (beat serializer between the arbiter and the bridge)
    s.config(2)
    case_regs(s, 2)
    case_b(s, 2, ideal=True, tag="nol2_ideal")
    case_c(s, 2)
    case_g(s, 2)
    case_h(s, 2)
    # configuration 3: no DMA (bridge drives m_axi directly)
    s.config(3)
    case_regs(s, 3)
    case_b(s, 3, ideal=False, tag="nodma_random")
    case_e(s, 3)
    case_f(s, 3)
    case_i(s, 3)
    return s


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the committed file is up to date instead of writing it")
    args = parser.parse_args(argv)
    text = build_script().render()
    if args.check:
        if OUTPUT.read_text(encoding="ascii") != text:
            print(f"{OUTPUT} is stale; rerun the generator", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
