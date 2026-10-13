"""Generate command programs for ``tb_npu_vector_unit`` (issue #136).

Outputs (committed): ``vector_unit_programs.txt`` for the VLEN = 64 instance
and ``vector_unit_programs_v128.txt``, a smaller set for the VLEN = 128
instance of the same testbench. Every program is run on a fresh TCM image and
compared with :class:`src.test.model.hx_vector.VectorUnitModel` after its
last command: the full register file, every TCM word the program touches, and
the sticky ``illegal`` flag.

File format (all numbers hex unless noted)::

    # comment line
    <programs>                                      (decimal)
    then per program:
    <name> <mode> <words> <commands> <perf_start>   (decimal)
    <words> lines:     <tcm_addr> <initial VLEN-bit word>
    <commands> lines:  <op> <type> <vd> <vs1> <vs2> <addr> <scalar> <pre>
    8 lines:           <expected register v0..v7, VLEN-bit>
    <words> lines:     <tcm_addr> <expected VLEN-bit word>
    <expected illegal> (0/1)

``mode`` bits: 1 random cmd_valid gaps, 2 random tcm_gnt stalls, 4 report
cycles per command from command ``perf_start`` on (performance program),
8 also used for the reset-in-flight test, 16 paced kernel (below). Every
program first loads all eight registers from its TCM words (VLD), so no
program depends on state left by another. Programs only address the TCM
words they list.

Paced kernels (mode 16) model the issuing scalar core: ``npu_scalar_core``
takes 3 cycles per instruction, so command n is presented 3 * (pre + 1)
cycles after command n - 1 was accepted, ``pre`` being the number of scalar
instructions (pointer bumps, loop branch) between them. The testbench lets
the register-initialisation VLDs drain, then reports the cycles from
presenting command ``perf_start`` until busy falls (the final VFENCE). ``pre``
is 0 in every other program.

Run from the repository root:
``python3 -m src.test.vectors.generate_vector_unit_vectors``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

from src.test.model import hx_vector as hv
from src.test.model.hx_vector import VectorCommand as Cmd
from src.test.vectors.generate_hx_numeric_vectors import FP16_CORNERS

DEFAULT_SEED = 0x136E
OUTPUT_FILE = "vector_unit_programs.txt"
OUTPUT_FILE_V128 = "vector_unit_programs_v128.txt"
TCM_WORDS = 1024
VLEN = 64
MAX_WORDS = 256

MODE_GAPS = 1
MODE_GNT = 2
MODE_PERF = 4
MODE_RESET = 8
MODE_PACED = 16

INT_CORNERS = {
    hv.TYPE_INT8: (0x00, 0x01, 0xFF, 0x7F, 0x80, 0x7E, 0x81, 0x02, 0x40, 0xC0, 0x03, 0xFD),
    hv.TYPE_INT16: (0x0000, 0x0001, 0xFFFF, 0x7FFF, 0x8000, 0x7FFE, 0x8001, 0x0100,
                    0x00FF, 0x4000, 0xC000, 0x0003, 0xFFFD, 0x1234),
    hv.TYPE_INT32: (0x00000000, 0x00000001, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000,
                    0x7FFFFFFE, 0x80000001, 0x00010000, 0x0000FFFF, 0x40000000,
                    0xC0000000, 0x00000003, 0xFFFFFFFD, 0x12345678, 0xFFFF0000,
                    0x00008000, 0xFFFF8000),
}

# Shift amounts for VSRA: 0, 1, lane-1, lane, lane+1, 31 for every lane width.
SRA_SHIFTS = (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 30, 31)


def pack(lanes, bits):
    return hv.join_lanes(list(lanes), bits)


class Program:
    def __init__(self, name: str, mode: int, rng: random.Random, vlen: int = VLEN):
        self.name = name
        self.mode = mode
        self.rng = rng
        self.vlen = vlen
        self.words: Dict[int, int] = {}
        self.commands: List[Cmd] = []
        self.pre: List[int] = []       # scalar instructions before each command
        self.perf_start = 0
        self._next = 0

    def word(self, value: Optional[int] = None, addr: Optional[int] = None) -> int:
        """Allocate a TCM word (optionally at a fixed address)."""
        if addr is None:
            while self._next in self.words:
                self._next += 1
            addr = self._next
        if addr in self.words:
            raise ValueError("address already allocated")
        if value is None:
            value = self.rng.getrandbits(self.vlen)
        self.words[addr] = value
        return addr

    def free_words(self) -> int:
        return MAX_WORDS - len(self.words)

    def init_registers(self, values=None) -> None:
        for r in range(8):
            v = values[r] if values is not None else random_word(self.rng, self.vlen)
            self.commands.append(Cmd(hv.OP_VLD, self.rng.randrange(4), r, 0, 0, self.word(v),
                                     self.rng.getrandbits(32)))
            self.pre.append(0)

    def add(self, op, typ=0, vd=0, vs1=0, vs2=0, addr=0, scalar=None, pre=0) -> None:
        if scalar is None:
            scalar = self.rng.getrandbits(32)
        self.commands.append(Cmd(op, typ, vd, vs1, vs2, addr, scalar))
        self.pre.append(pre)

    def render(self) -> List[str]:
        model = hv.VectorUnitModel(vlen=self.vlen, tcm_words=TCM_WORDS)
        for addr, value in self.words.items():
            model.tcm[addr] = value
        for cmd in self.commands:
            if cmd.op in (hv.OP_VLD, hv.OP_VST) and hv.is_legal(cmd.op, cmd.typ):
                assert cmd.addr in self.words, f"{self.name}: unlisted address {cmd.addr}"
        model.run(self.commands)
        addrs = sorted(self.words)
        nd = self.vlen // 4
        lines = [f"{self.name} {self.mode} {len(addrs)} {len(self.commands)} {self.perf_start}"]
        lines += [f"{a:03x} {self.words[a]:0{nd}x}" for a in addrs]
        lines += [f"{c.op:02x} {c.typ:x} {c.vd:x} {c.vs1:x} {c.vs2:x} {c.addr:03x} {c.scalar:08x} "
                  f"{pre:x}" for c, pre in zip(self.commands, self.pre)]
        lines += [f"{v:0{nd}x}" for v in model.rf]
        lines += [f"{a:03x} {model.tcm[a]:0{nd}x}" for a in addrs]
        lines.append("1" if model.illegal else "0")
        return lines


# --------------------------------------------------------------------------
# operand data
# --------------------------------------------------------------------------
def random_fp16(rng: random.Random) -> int:
    pick = rng.random()
    if pick < 0.3:
        return rng.choice(FP16_CORNERS)
    if pick < 0.8:  # moderate magnitudes: meaningful sums and products
        return (rng.getrandbits(1) << 15) | (rng.randrange(8, 23) << 10) | rng.getrandbits(10)
    return rng.getrandbits(16)


def random_word(rng: random.Random, vlen: int = VLEN) -> int:
    pick = rng.randrange(6)
    if pick == 0:
        return rng.getrandbits(vlen)
    if pick == 4:
        return pack((random_fp16(rng) for _ in range(vlen // 16)), 16)
    if pick == 5:  # small integers (no saturation, exact products)
        bits = rng.choice((8, 16, 32))
        return pack((rng.randrange(-20, 20) for _ in range(vlen // bits)), bits)
    typ = (hv.TYPE_INT8, hv.TYPE_INT16, hv.TYPE_INT32)[pick - 1]
    bits = hv.LANE_BITS[typ]
    return pack((rng.choice(INT_CORNERS[typ]) if rng.random() < 0.6 else rng.getrandbits(bits)
                 for _ in range(vlen // bits)), bits)


def corner_operand_words(typ: int, rng: random.Random, vlen: int = VLEN):
    """Pairs of words whose lanes cover the cross product of the corners."""
    corners = FP16_CORNERS if typ == hv.TYPE_FP16 else INT_CORNERS[typ]
    # second operands: every corner for INT8/INT16, a subset elsewhere (the
    # FP16 units have their own exhaustive corner tests)
    partners = corners if len(corners) <= 14 else corners[::2]
    bits = hv.LANE_BITS[typ]
    lanes = vlen // bits
    pairs = [(x, y) for x in corners for y in partners]
    while len(pairs) % lanes:
        pairs.append((rng.getrandbits(bits), rng.getrandbits(bits)))
    words = []
    for i in range(0, len(pairs), lanes):
        chunk = pairs[i:i + lanes]
        words.append((pack((x for x, _ in chunk), bits), pack((y for _, y in chunk), bits)))
    return words


# --------------------------------------------------------------------------
# directed programs
# --------------------------------------------------------------------------
def directed_arith(rng: random.Random, vlen: int = VLEN) -> List[Program]:
    """Every arithmetic op x type over the corner cross products."""
    programs = []
    ops = (hv.OP_VADD, hv.OP_VSUB, hv.OP_VMUL, hv.OP_VMAX, hv.OP_VMIN, hv.OP_VRELU, hv.OP_VMOV)
    for typ in (hv.TYPE_INT8, hv.TYPE_INT16, hv.TYPE_INT32, hv.TYPE_FP16):
        words = corner_operand_words(typ, rng, vlen)
        prog = None
        for a, b in words:
            if prog is None or prog.free_words() < 2 + len(ops):
                prog = Program(f"dir_arith_{hv.TYPE_NAMES[typ]}_{len(programs)}", 0, rng, vlen)
                prog.init_registers()
                programs.append(prog)
            prog.add(hv.OP_VLD, typ, 1, addr=prog.word(a))
            prog.add(hv.OP_VLD, typ, 2, addr=prog.word(b))
            dests = (3, 4, 5, 6, 7, 0, 3)
            for op, vd in zip(ops, dests):
                prog.add(op, typ, vd, 1, 2)
                prog.add(hv.OP_VST, typ, 0, vd, 0, addr=prog.word())
    return programs


def directed_sra(rng: random.Random, vlen: int = VLEN) -> List[Program]:
    programs = []
    for typ in (hv.TYPE_INT8, hv.TYPE_INT16, hv.TYPE_INT32):
        bits = hv.LANE_BITS[typ]
        corners = list(INT_CORNERS[typ])
        # rounding corners: odd values, exact halves, +-1, extremes
        lanes = vlen // bits
        values = corners + [3, (1 << bits) - 3, 5, (1 << bits) - 5, 6, (1 << bits) - 6]
        while len(values) % lanes:
            values.append(rng.getrandbits(bits))
        words = [pack(values[i:i + lanes], bits) for i in range(0, len(values), lanes)]
        words += [random_word(rng, vlen) for _ in range(4)]
        prog = Program(f"dir_sra_{hv.TYPE_NAMES[typ]}", 0, rng, vlen)
        prog.init_registers()
        for w in words:
            prog.add(hv.OP_VLD, typ, 1, addr=prog.word(w))
            for k, s in enumerate(SRA_SHIFTS):
                vd = 2 + (k % 6)
                # upper scalar bits are ignored; vary them
                prog.add(hv.OP_VSRA, typ, vd, 1, rng.randrange(8),
                         scalar=(rng.getrandbits(27) << 5) | s)
                prog.add(hv.OP_VST, typ, 0, vd, 0, addr=prog.word())
        programs.append(prog)
    return programs


def directed_splat_mov(rng: random.Random, vlen: int = VLEN) -> Program:
    prog = Program("dir_splat_mov_addr", 0, rng, vlen)
    prog.init_registers()
    scalars = [0, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x12345678, 0x0000807F, 0x00007E00,
               0xFFFF8000, 0x00008000]
    for typ in range(4):
        for s in scalars:
            prog.add(hv.OP_VSPLAT, typ, 3, rng.randrange(8), rng.randrange(8), scalar=s)
            prog.add(hv.OP_VST, typ, 0, 3, 0, addr=prog.word())
    # VMOV chains and VMOV onto itself
    prog.add(hv.OP_VMOV, 0, 4, 3)
    prog.add(hv.OP_VMOV, 1, 5, 4)
    prog.add(hv.OP_VMOV, 2, 5, 5)
    prog.add(hv.OP_VST, 3, 0, 5, 0, addr=prog.word())
    # full TCM address width: lowest, highest and alternating-bit addresses
    for addr in (0x3FF, 0x200, 0x1FF, 0x2AA, 0x155, 0x3FE):
        prog.word(addr=addr)
        prog.add(hv.OP_VLD, 0, 6, addr=addr)
        prog.add(hv.OP_VST, 0, 0, 6, 0, addr=0x3FF - addr if (0x3FF - addr) in prog.words
                 else addr)
    return prog


def directed_hazards(rng: random.Random, vlen: int = VLEN) -> List[Program]:
    programs = []
    for typ in range(4):
        prog = Program(f"dir_hazard_{hv.TYPE_NAMES[typ]}", 0, rng, vlen)
        prog.init_registers()
        a, b, c = prog.word(), prog.word(), prog.word()
        # RAW + WAW chain on one register, every arithmetic op
        for op in (hv.OP_VADD, hv.OP_VMUL, hv.OP_VSUB, hv.OP_VMAX, hv.OP_VMIN, hv.OP_VRELU,
                   hv.OP_VADD, hv.OP_VMUL):
            prog.add(op, typ, 1, 1, 2)
        prog.add(hv.OP_VST, typ, 0, 1, 0, addr=a)
        # VST then VLD of the same address, then a dependent op, then VST
        prog.add(hv.OP_VLD, typ, 3, addr=a)
        prog.add(hv.OP_VADD, typ, 4, 3, 3)
        prog.add(hv.OP_VST, typ, 0, 4, 0, addr=b)
        prog.add(hv.OP_VLD, typ, 5, addr=b)
        # WAR: read v5 then overwrite it with a short-latency op
        prog.add(hv.OP_VMUL, typ, 6, 5, 2)
        prog.add(hv.OP_VSPLAT, typ, 5, scalar=0x3C003C00)
        # WAW: long-latency op then short op to the same register
        prog.add(hv.OP_VADD, typ, 7, 5, 6)
        prog.add(hv.OP_VMOV, typ, 7, 2)
        prog.add(hv.OP_VST, typ, 0, 7, 0, addr=c)
        # back-to-back stores to one address: the last wins; then load it
        prog.add(hv.OP_VST, typ, 0, 1, 0, addr=c)
        prog.add(hv.OP_VST, typ, 0, 6, 0, addr=c)
        prog.add(hv.OP_VLD, typ, 0, addr=c)
        prog.add(hv.OP_VLD, typ, 1, addr=a)
        prog.add(hv.OP_VST, typ, 0, 0, 0, addr=a)
        prog.add(hv.OP_VST, typ, 0, 1, 0, addr=b)
        programs.append(prog)
    return programs


def directed_illegal(rng: random.Random, vlen: int = VLEN) -> List[Program]:
    prog = Program("dir_illegal", 0, rng, vlen)
    prog.init_registers()
    out = prog.word()
    for op in range(11, 32):
        prog.add(op, rng.randrange(4), 1, 2, 3, addr=out)  # would write v1 / TCM
        prog.add(hv.OP_VADD, hv.TYPE_INT8, 2, 2, 3)
    prog.add(hv.OP_VSRA, hv.TYPE_FP16, 4, 4, 0, scalar=1)
    prog.add(hv.OP_VST, 0, 0, 1, 0, addr=out)
    clean = Program("dir_legal_only", 0, rng, vlen)
    clean.init_registers()
    for typ in range(3):
        clean.add(hv.OP_VSRA, typ, 1, 2, scalar=3)
    clean.add(hv.OP_VST, 0, 0, 1, 0, addr=clean.word())
    return [prog, clean]


# --------------------------------------------------------------------------
# performance programs (no gaps, no stalls): independent and dependent
# --------------------------------------------------------------------------
PERF_N = 32


def perf_programs(rng: random.Random) -> List[Program]:
    programs = []
    cases = [(hv.OP_VLD, 0), (hv.OP_VST, 0), (hv.OP_VSPLAT, 0), (hv.OP_VMOV, 0)]
    for op in (hv.OP_VADD, hv.OP_VSUB, hv.OP_VMUL, hv.OP_VMAX, hv.OP_VMIN, hv.OP_VRELU):
        cases += [(op, t) for t in range(4)]
    cases += [(hv.OP_VSRA, t) for t in range(3)]
    for op, typ in cases:
        for dep in (False, True):
            if op == hv.OP_VSPLAT and dep:
                continue
            kind = "dep" if dep else "indep"
            prog = Program(f"perf_{kind}_{hv.OP_NAMES[op]}_{hv.TYPE_NAMES[typ]}", MODE_PERF, rng)
            prog.init_registers()
            prog.perf_start = len(prog.commands)
            addrs = [prog.word() for _ in range(8)]
            for i in range(PERF_N):
                if op in (hv.OP_VLD, hv.OP_VST) and dep:
                    # store, then load the same word back into the stored register
                    if i % 2 == 0:
                        prog.add(hv.OP_VST, typ, 0, 2, 0, addr=addrs[0])
                    else:
                        prog.add(hv.OP_VLD, typ, 2, addr=addrs[0])
                elif op == hv.OP_VLD:
                    prog.add(op, typ, 2 + i % 6, addr=addrs[i % 8])
                elif op == hv.OP_VST:
                    prog.add(op, typ, 0, i % 2, addr=addrs[i % 8])
                elif dep:
                    prog.add(op, typ, 2, 2, 1, scalar=1)
                else:
                    prog.add(op, typ, 2 + i % 6, 0, 1, scalar=1)
            programs.append(prog)
    return programs


# --------------------------------------------------------------------------
# paced kernels: end-to-end cycles of typical elementwise loops as the
# scalar core issues them (3 cycles per instruction, ``pre`` = scalar
# instructions before a command). Three codings of each kernel:
#   loop   : rolled loop, pointer bumps + counter + branch after the store
#   unroll : straight-line, addresses already in scalar registers (the
#            densest command stream the core can send)
#   sched  : straight-line and software-pipelined so that no command reads
#            the result of the command right before it
# --------------------------------------------------------------------------
RES_WORDS = 8      # 64 INT8 elements at VLEN = 64
AXPY_WORDS = 16    # 64 FP16 elements at VLEN = 64


def kernel_residual(rng: random.Random, coding: str) -> Program:
    """c[i] = sat(a[i] + b[i]), INT8: vld a; vld b; vadd; vst."""
    prog = Program(f"kern_res8_{coding}", MODE_PERF | MODE_PACED, rng)
    prog.init_registers()
    prog.perf_start = len(prog.commands)
    a = [prog.word(random_word(rng)) for _ in range(RES_WORDS)]
    b = [prog.word(random_word(rng)) for _ in range(RES_WORDS)]
    c = [prog.word() for _ in range(RES_WORDS)]
    i8 = hv.TYPE_INT8
    if coding == "sched":
        sets = ((1, 2, 3), (4, 5, 6))
        prog.add(hv.OP_VLD, i8, 1, addr=a[0])
        prog.add(hv.OP_VLD, i8, 2, addr=b[0])
        for i in range(RES_WORDS):
            ra, rb, rc = sets[i % 2]
            prog.add(hv.OP_VADD, i8, rc, ra, rb)
            if i + 1 < RES_WORDS:
                na, nb, _ = sets[(i + 1) % 2]
                prog.add(hv.OP_VLD, i8, na, addr=a[i + 1])
                prog.add(hv.OP_VLD, i8, nb, addr=b[i + 1])
            prog.add(hv.OP_VST, i8, 0, rc, 0, addr=c[i])
        return prog
    overhead = 5 if coding == "loop" else 0   # 3 pointer addi, counter addi, bne
    for i in range(RES_WORDS):
        prog.add(hv.OP_VLD, i8, 1, addr=a[i], pre=overhead if i else 0)
        prog.add(hv.OP_VLD, i8, 2, addr=b[i])
        prog.add(hv.OP_VADD, i8, 3, 1, 2)
        prog.add(hv.OP_VST, i8, 0, 3, 0, addr=c[i])
    return prog


def kernel_axpy(rng: random.Random, coding: str) -> Program:
    """y[i] = alpha * x[i] + y[i], FP16 (in place): vld x; vld y; vmul; vadd; vst."""
    prog = Program(f"kern_axpy16_{coding}", MODE_PERF | MODE_PACED, rng)
    prog.init_registers()
    prog.perf_start = len(prog.commands)
    fp = hv.TYPE_FP16
    x = [prog.word(pack((random_fp16(rng) for _ in range(VLEN // 16)), 16))
         for _ in range(AXPY_WORDS)]
    y = [prog.word(pack((random_fp16(rng) for _ in range(VLEN // 16)), 16))
         for _ in range(AXPY_WORDS)]
    prog.add(hv.OP_VSPLAT, fp, 7, scalar=0x3E00)   # alpha = 1.5
    if coding == "sched":
        # x / t share one register, y / result another; two register sets
        sets = ((1, 2), (3, 4))
        prog.add(hv.OP_VLD, fp, 1, addr=x[0])
        prog.add(hv.OP_VLD, fp, 2, addr=y[0])
        prog.add(hv.OP_VMUL, fp, 1, 7, 1)
        if AXPY_WORDS > 1:
            prog.add(hv.OP_VLD, fp, 3, addr=x[1])
            prog.add(hv.OP_VLD, fp, 4, addr=y[1])
        for i in range(AXPY_WORDS):
            rx, ry = sets[i % 2]
            prog.add(hv.OP_VADD, fp, ry, rx, ry)
            if i + 1 < AXPY_WORDS:
                nx, _ = sets[(i + 1) % 2]
                prog.add(hv.OP_VMUL, fp, nx, 7, nx)
            prog.add(hv.OP_VST, fp, 0, ry, 0, addr=y[i])
            if i + 2 < AXPY_WORDS:
                prog.add(hv.OP_VLD, fp, rx, addr=x[i + 2])
                prog.add(hv.OP_VLD, fp, ry, addr=y[i + 2])
        return prog
    overhead = 4 if coding == "loop" else 0   # 2 pointer addi, counter addi, bne
    for i in range(AXPY_WORDS):
        prog.add(hv.OP_VLD, fp, 1, addr=x[i], pre=overhead if i else 0)
        prog.add(hv.OP_VLD, fp, 2, addr=y[i])
        prog.add(hv.OP_VMUL, fp, 3, 7, 1)
        prog.add(hv.OP_VADD, fp, 4, 3, 2)
        prog.add(hv.OP_VST, fp, 0, 4, 0, addr=y[i])
    return prog


def kernel_programs(rng: random.Random) -> List[Program]:
    programs = []
    for coding in ("loop", "unroll", "sched"):
        programs.append(kernel_residual(rng, coding))
        programs.append(kernel_axpy(rng, coding))
    return programs


# --------------------------------------------------------------------------
# random programs
# --------------------------------------------------------------------------
RANDOM_OPS = ([hv.OP_VLD] * 12 + [hv.OP_VST] * 12 + [hv.OP_VSPLAT] * 5 +
              [hv.OP_VADD] * 9 + [hv.OP_VSUB] * 7 + [hv.OP_VMUL] * 9 + [hv.OP_VMAX] * 6 +
              [hv.OP_VMIN] * 6 + [hv.OP_VRELU] * 5 + [hv.OP_VSRA] * 8 + [hv.OP_VMOV] * 5)


def random_program(rng: random.Random, name: str, mode: int, count: int,
                   illegal_rate: float, vlen: int = VLEN) -> Program:
    prog = Program(name, mode, rng, vlen)
    # scattered addresses over the whole TCM, few enough to alias often
    addrs = rng.sample(range(TCM_WORDS), 24)
    if rng.random() < 0.5:
        addrs[0], addrs[1] = 0, TCM_WORDS - 1
    for a in addrs:
        prog.word(random_word(rng, vlen), addr=a)
    for r in range(8):
        prog.add(hv.OP_VLD, rng.randrange(4), r, addr=addrs[r])
    last_vd, last_addr = 0, addrs[0]
    for _ in range(count):
        if rng.random() < illegal_rate:
            op = rng.randrange(11, 32)
        else:
            op = rng.choice(RANDOM_OPS)
        typ = rng.randrange(4)
        vd = rng.randrange(8)
        vs1 = last_vd if rng.random() < 0.4 else rng.randrange(8)
        vs2 = last_vd if rng.random() < 0.3 else rng.randrange(8)
        addr = last_addr if rng.random() < 0.3 else rng.choice(addrs)
        if op == hv.OP_VSRA:
            scalar = (rng.getrandbits(27) << 5) | (rng.choice(SRA_SHIFTS) if rng.random() < 0.6
                                                   else rng.randrange(32))
        elif op == hv.OP_VSPLAT:
            scalar = (pack([random_fp16(rng)] * 2, 16) if typ == hv.TYPE_FP16
                      else rng.choice((rng.getrandbits(32), random_word(rng, vlen) & 0xFFFFFFFF)))
        else:
            scalar = rng.getrandbits(32)
        prog.add(op, typ, vd, vs1, vs2, addr, scalar)
        if hv.is_legal(op, typ) and op != hv.OP_VST:
            last_vd = vd
        if op in (hv.OP_VLD, hv.OP_VST):
            last_addr = addr
    return prog


def random_programs(rng: random.Random) -> List[Program]:
    programs = []
    for i in range(10):
        mode = (MODE_GAPS | MODE_GNT) if i % 3 else MODE_GAPS
        if i == 1:
            mode |= MODE_RESET
        programs.append(random_program(rng, f"rand_{i}", mode, 300, 0.02))
    programs.append(random_program(rng, "rand_legal_0", MODE_GAPS | MODE_GNT, 300, 0.0))
    programs.append(random_program(rng, "rand_legal_1", 0, 300, 0.0))
    programs.append(random_program(rng, "rand_long_nogap", 0, 1500, 0.01))
    programs.append(random_program(rng, "rand_long_stall", MODE_GNT, 1000, 0.01))
    return programs


def build(seed: int = DEFAULT_SEED) -> List[Program]:
    rng = random.Random(seed)
    programs = []
    programs += directed_arith(rng)
    programs += directed_sra(rng)
    programs.append(directed_splat_mov(rng))
    programs += directed_hazards(rng)
    programs += directed_illegal(rng)
    programs += perf_programs(rng)
    programs += random_programs(rng)
    programs += kernel_programs(rng)
    return programs


def build_v128(seed: int = DEFAULT_SEED) -> List[Program]:
    """Smaller set for the VLEN = 128 instance (four slices per register):
    the corner cross products of every op x type, SRA, VSPLAT / VMOV / full
    address width, hazards, illegal commands and random programs with gaps,
    grant stalls and a reset in flight."""
    rng = random.Random(seed + 128)
    v = 128
    programs = []
    programs += directed_arith(rng, v)
    programs += directed_sra(rng, v)
    programs.append(directed_splat_mov(rng, v))
    programs += directed_hazards(rng, v)
    programs += directed_illegal(rng, v)
    programs.append(random_program(rng, "rand_v128_0", MODE_GAPS | MODE_GNT | MODE_RESET, 300,
                                   0.02, v))
    programs.append(random_program(rng, "rand_v128_1", 0, 400, 0.01, v))
    programs.append(random_program(rng, "rand_v128_2", MODE_GNT, 400, 0.01, v))
    return programs


def render_file(programs: List[Program], seed: int, vlen: int) -> str:
    commands = sum(len(p.commands) for p in programs)
    lines = [f"# npu_vector_unit programs vlen={vlen} seed={seed}: {len(programs)} programs, "
             f"{commands} commands (generate_vector_unit_vectors.py)",
             str(len(programs))]
    for prog in programs:
        lines += prog.render()
    return "\n".join(lines) + "\n"


def write_all(directory: Path, seed: int = DEFAULT_SEED) -> None:
    (directory / OUTPUT_FILE).write_text(render_file(build(seed), seed, VLEN),
                                         encoding="ascii", newline="\n")
    (directory / OUTPUT_FILE_V128).write_text(render_file(build_v128(seed), seed, 128),
                                              encoding="ascii", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    write_all(arguments.output_dir, arguments.seed)


if __name__ == "__main__":
    main()
