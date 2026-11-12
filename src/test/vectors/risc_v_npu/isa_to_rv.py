"""Translate the 64-bit NPU ISA test programs into RISC-V programs for CVA6.

Reads src/test/vectors/isa_frontend_16x16.txt (written by
generate_isa_vectors.py: per case a memory image holding the ISA program at 0
and the data region at DATA_BASE, plus the golden image from isa_sim.py),
replaces each ISA program by an equivalent RV32I program assembled with
rv32asm.py, moves the data region behind it, and writes the same text format
for tb_scalar_core_cva6:

    case <words> <prog_len> <data_base> <error> <jobs> <name>
    <words> 64-bit hex words: initial image (RISC-V binary at 0, data at data_base)
    <words> 64-bit hex words: expected final image
    ...
    end 0 0 0 0 0 none

prog_len is the binary size in 64-bit words (rounded up); <error> and <jobs>
are those of the ISA case.

    python -m src.test.vectors.risc_v_npu.isa_to_rv [-o OUT] [--summary]

The generated program (MMIO register map: cva6_npu_mmio.sv, base 0x4000_0000)
uses registers only, never the stack or any DDR data, so the memory image
outside the C tiles stays as loaded and can be compared word for word:

    s1 = MMIO base               s0 = DATA_BASE (read from the MMIO register)
    s2, s3, s4 = absolute A/B/C addresses (DATA_BASE + ISA offset)
    s5, s6, s7 = A/B/C increments  t0 = scratch   t1 = REPEAT counter

Mapping of the ISA (src/vendors/npu_pynq/spec/isa.py, isa_sim.py):

    SHAPE m,n,k     sw SHAPE_MN, sw SHAPE_K (skipped if the same shape is
                    already in the registers)
    ADDR_A/B/C off  s2/s3/s4 = s0 + off, stored to ADDR_A/B/C at once
    INCR            s5..s7 (registers only)
    GEMM flags      sw DOORBELL; then each flagged address is incremented and
                    stored, ready for the next job
    REPEAT n; GEMM  a counted loop around the GEMM sequence; REPEAT of any
                    other instruction emits it once (they are idempotent)
    FENCE           lw FENCE (the read returns once every job is written back)
    NOP             nothing
    END             lw FENCE; sw DONE = 0; spin
    illegal opcode  no RISC-V equivalent (the program is translated ahead of
                    time, so the translator sees it): lw FENCE; sw DONE =
                    ERR_ILLEGAL_OPCODE (0x10); spin. This reproduces the CUSTOM
                    core's result: earlier jobs retire, ISA_ERROR = 0x10.
    no END          sw DONE = ERR_FETCH (0x15)

A bad shape or a misaligned address is not checked in software: the DOORBELL
latches ERR_BAD_SHAPE / ERR_MISALIGNED in cva6_npu_mmio, which ends the
program with the same code as the CUSTOM core.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct

from src.vendors.npu_pynq.spec import isa
from src.test.vectors.risc_v_npu.rv32asm import Assembler

SOURCE = Path(__file__).resolve().parent.parent / "isa_frontend_16x16.txt"

MMIO_BASE = 0x4000_0000
REG_SHAPE_MN = 0x00
REG_SHAPE_K = 0x04
REG_ADDR = {"a": 0x08, "b": 0x0C, "c": 0x10}
REG_DOORBELL = 0x14
REG_STATUS = 0x18
REG_FENCE = 0x1C
REG_DONE = 0x20
REG_DATA_BASE = 0x24

ADDR_REG = {"a": "s2", "b": "s3", "c": "s4"}
INC_REG = {"a": "s5", "b": "s6", "c": "s7"}
FLAG = {"a": isa.GEMM_INC_A, "b": isa.GEMM_INC_B, "c": isa.GEMM_INC_C}


@dataclass
class Case:
    name: str
    image: list[int]      # 64-bit words
    expected: list[int]
    prog_len: int         # in 64-bit words
    data_base: int        # bytes
    error: int
    jobs: int


def read_cases(path: Path = SOURCE) -> list[Case]:
    lines = path.read_text().splitlines()
    cases: list[Case] = []
    i = 1  # header comment
    while True:
        tag, words, prog_len, data_base, error, jobs, name = lines[i].split()
        i += 1
        words = int(words)
        if tag == "end" or words == 0:
            return cases
        image = [int(x, 16) for x in lines[i:i + words]]
        expected = [int(x, 16) for x in lines[i + words:i + 2 * words]]
        i += 2 * words
        cases.append(Case(name, image, expected, int(prog_len), int(data_base), int(error), int(jobs)))


class _Translator:
    def __init__(self) -> None:
        self.asm = Assembler()
        self.shape: tuple[int, int] | None = None   # (SHAPE_MN, SHAPE_K) in the MMIO registers
        self.synced = {x: False for x in "abc"}       # ADDR_x register equals sX
        self.loops = 0

    def store_addr(self, x: str) -> None:
        self.asm.sw(ADDR_REG[x], REG_ADDR[x], "s1")
        self.synced[x] = True

    def gemm(self, flags: int) -> None:
        a = self.asm
        if self.shape is None:
            # The ISA starts with m = n = k = 0; the MMIO registers keep the
            # previous run's shape, so make a GEMM before any SHAPE fault too.
            a.sw("zero", REG_SHAPE_MN, "s1")
            a.sw("zero", REG_SHAPE_K, "s1")
            self.shape = (0, 0)
        for x in "abc":
            if not self.synced[x]:
                self.store_addr(x)
        a.sw("zero", REG_DOORBELL, "s1")
        for x in "abc":
            if flags & FLAG[x]:
                a.add(ADDR_REG[x], ADDR_REG[x], INC_REG[x])
                self.store_addr(x)

    def finish(self, code: int) -> None:
        a = self.asm
        a.lw("t0", REG_FENCE, "s1")
        if code:
            a.li("t0", code)
            a.sw("t0", REG_DONE, "s1")
        else:
            a.sw("zero", REG_DONE, "s1")
        spin = f"spin{self.loops}"
        self.loops += 1
        a.label(spin)
        a.j(spin)

    def one(self, inst: isa.Instruction) -> bool:
        """Emit one ISA instruction; False once the program has ended."""
        a = self.asm
        op, f = inst.opcode, inst.fields()
        if op in (isa.OP_NOP,):
            pass
        elif op == isa.OP_FENCE:
            a.lw("t0", REG_FENCE, "s1")
        elif op == isa.OP_END:
            self.finish(0)
            return False
        elif op == isa.OP_SHAPE:
            want = ((f["n"] << 16) | f["m"], f["k"])
            if self.shape is None or self.shape[0] != want[0]:
                a.li("t0", want[0])
                a.sw("t0", REG_SHAPE_MN, "s1")
            if self.shape is None or self.shape[1] != want[1]:
                a.li("t0", want[1])
                a.sw("t0", REG_SHAPE_K, "s1")
            self.shape = want
        elif op in (isa.OP_ADDR_A, isa.OP_ADDR_B, isa.OP_ADDR_C):
            x = {isa.OP_ADDR_A: "a", isa.OP_ADDR_B: "b", isa.OP_ADDR_C: "c"}[op]
            a.li("t0", f["offset"])
            a.add(ADDR_REG[x], "s0", "t0")
            self.store_addr(x)
        elif op == isa.OP_INCR:
            for x in "abc":
                a.li(INC_REG[x], f[f"{x}_inc"])
        elif op == isa.OP_GEMM:
            self.gemm(f["flags"])
        else:
            self.finish(isa.ERR_ILLEGAL_OPCODE)
            return False
        return True


def translate(words: list[int]) -> bytes:
    """RV32I machine code (little-endian) equivalent to the ISA program."""
    t = _Translator()
    a = t.asm
    a.lui("s1", MMIO_BASE >> 12)
    a.lw("s0", REG_DATA_BASE, "s1")
    for x in "abc":
        a.mv(ADDR_REG[x], "s0")
        a.li(INC_REG[x], 0)
    pc = 0
    while pc < len(words):
        inst = isa.decode(words[pc])
        if inst.opcode == isa.OP_REPEAT:
            count = inst.fields()["count"]
            if pc + 1 >= len(words):
                break
            pc += 1
            body = isa.decode(words[pc])
            if body.opcode == isa.OP_GEMM and count > 1:
                # All three addresses are in their registers before the loop,
                # so the loop body needs no stores but the flagged ones.
                for x in "abc":
                    if not t.synced[x]:
                        t.store_addr(x)
                if t.shape is None:
                    a.sw("zero", REG_SHAPE_MN, "s1")
                    a.sw("zero", REG_SHAPE_K, "s1")
                    t.shape = (0, 0)
                loop = f"rep{t.loops}"
                t.loops += 1
                a.li("t1", count)
                a.label(loop)
                t.gemm(body.fields()["flags"])
                a.addi("t1", "t1", -1)
                a.bnez("t1", loop)
            elif not t.one(body):
                break
        elif not t.one(inst):
            break
        pc += 1
    else:
        t.finish(isa.ERR_FETCH)
    return a.assemble()


def relocate(case: Case, code: bytes) -> Case:
    """The case with its ISA program replaced by ``code`` and the data moved."""
    code = code + b"\0" * (-len(code) % 8)
    prog = [w for (w,) in struct.iter_unpack("<Q", code)]
    data_base = isa.align(len(code) + 8, 256)
    old = case.data_base // 8
    pad = [0] * (data_base // 8 - len(prog))
    image = prog + pad + case.image[old:]
    expected = prog + pad + case.expected[old:]
    return Case(case.name, image, expected, len(prog), data_base, case.error, case.jobs)


def render(cases: list[Case]) -> str:
    lines = ["# generated by src/test/vectors/risc_v_npu/isa_to_rv.py from isa_frontend_16x16.txt"]
    for c in cases:
        lines.append(f"case {len(c.image)} {c.prog_len} {c.data_base} {c.error} {c.jobs} {c.name}")
        lines.extend(f"{w:016x}" for w in c.image)
        lines.extend(f"{w:016x}" for w in c.expected)
    lines.append("end 0 0 0 0 0 none")
    return "\n".join(lines) + "\n"


def build(source: Path = SOURCE) -> list[tuple[Case, Case, int]]:
    """[(ISA case, RISC-V case, RISC-V code bytes)]"""
    out = []
    for case in read_cases(source):
        code = translate(case.image[:case.prog_len])
        out.append((case, relocate(case, code), len(code)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path, default=Path("build/cva6_frontend_16x16.txt"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--summary", action="store_true", help="print program sizes per case")
    args = parser.parse_args()
    built = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render([rv for _, rv, _ in built]))
    if args.summary:
        for isa_case, rv, size in built:
            print(f"{isa_case.name}: ISA {isa_case.prog_len} words ({isa_case.prog_len * 8} B), "
                  f"RISC-V {size // 4} instructions ({size} B), data_base 0x{rv.data_base:x}")
    print(f"wrote {args.output} ({len(built)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
