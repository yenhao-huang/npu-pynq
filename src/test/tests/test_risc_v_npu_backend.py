"""pynqo-backend's risc_v_npu delegate: the RISC-V programs issue the npu jobs.

One INT8 GEMM goes through pynqo-backend twice, as ``[cpu, npu]`` and as
``[cpu, risc_v_npu]``, each fed what pynqo-middle-end writes for that device
list (``write_model``: a CPU graph calling a bodiless ``@__delegate_0``, and
``delegates/<device>.mlir`` holding the subgraph). The npu ISA program runs on
the golden model (``src/test/model/isa_sim.py``); every risc_v_npu segment
binary runs on the small RV32IMC interpreter below, which records the jobs its
MMIO stores issue. Both must issue the same jobs, in the same order, at the
same offsets, and both delegate tables must give the subgraph's result size.

Skipped where pynqo-backend is not built (``src/export/pipeline.py``'s
``build_backend`` builds it; ``PYNQO_BACKEND_BUILD`` points at another build).
"""

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.export import pipeline
from src.vendors.npu_pynq.spec import isa
from src.test.model import isa_sim

REPO = Path(__file__).resolve().parents[3]

MMIO_BASE = 0x4000_0000
SHAPE_MN, SHAPE_K, ADDR_A, ADDR_B, ADDR_C, DOORBELL = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14
FENCE, DONE, DATA_BASE = 0x1C, 0x20, 0x24


def backend_tool() -> str:
    build = os.environ.get('PYNQO_BACKEND_BUILD') or pipeline._linux_path(REPO / 'build' / 'pynqo-backend')
    return f'{build}/pynqo-backend'


def tool_is_built() -> bool:
    try:
        pipeline._linux(f'test -x {backend_tool()}')
    except (OSError, RuntimeError):
        return False
    return True


# ------------------------------------------------------------ RV32IMC model

def _sext(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


class Halt(Exception):
    pass


class Rv32:
    """Enough of RV32IMC (machine mode, no interrupts) to run a program.

    Memory is the program image at ``base``; MMIO stores are recorded and
    turned into jobs on DOORBELL; DONE ends the run.
    """

    def __init__(self, image: bytes, mem_bytes: int, base: int, data_base: int):
        self.mem = bytearray(image) + bytearray(mem_bytes - len(image))
        self.base, self.data_base = base, data_base
        self.x = [0] * 32
        self.pc = base
        self.regs: dict[int, int] = {}
        self.jobs: list[tuple[int, int, int, int, int, int]] = []
        self.done: int | None = None
        self.csr: dict[int, int] = {}
        self.steps = 0

    # memory
    def load(self, addr: int, size: int, signed: bool) -> int:
        if addr >> 12 == MMIO_BASE >> 12:
            off = addr - MMIO_BASE
            if size != 4:
                raise AssertionError(f'MMIO load of {size} bytes at {addr:#x}')
            if off == DATA_BASE:
                return self.data_base
            if off == FENCE:
                return 0    # nothing outstanding, no error
            return self.regs.get(off, 0)
        i = addr - self.base
        if not 0 <= i <= len(self.mem) - size:
            raise AssertionError(f'load outside the program at {addr:#x}')
        v = int.from_bytes(self.mem[i:i + size], 'little')
        return _sext(v, 8 * size) & 0xFFFFFFFF if signed else v

    def store(self, addr: int, size: int, value: int) -> None:
        value &= (1 << (8 * size)) - 1
        if addr >> 12 == MMIO_BASE >> 12:
            off = addr - MMIO_BASE
            if size != 4:
                raise AssertionError(f'MMIO store of {size} bytes at {addr:#x}')
            if off == DOORBELL:
                mn = self.regs[SHAPE_MN]
                self.jobs.append((mn & 0xFFFF, mn >> 16, self.regs[SHAPE_K] & 0xFFFF,
                                  self.regs[ADDR_A], self.regs[ADDR_B], self.regs[ADDR_C]))
            elif off == DONE:
                self.done = value & 0xFF
                raise Halt
            else:
                self.regs[off] = value
            return
        i = addr - self.base
        if not 0 <= i <= len(self.mem) - size:
            raise AssertionError(f'store outside the program at {addr:#x}')
        self.mem[i:i + size] = value.to_bytes(size, 'little')

    def set(self, rd: int, value: int) -> None:
        if rd:
            self.x[rd] = value & 0xFFFFFFFF

    def run(self, budget: int = 1_000_000) -> None:
        try:
            while self.steps < budget:
                self.steps += 1
                lo = self.load(self.pc, 2, False)
                if lo & 3 == 3:
                    self.exec32(lo | self.load(self.pc + 2, 2, False) << 16)
                else:
                    self.exec16(lo)
        except Halt:
            return
        raise AssertionError('instruction budget exhausted')

    def alu(self, f3: int, a: int, b: int, alt: bool) -> int:
        sa, sb = _sext(a, 32), _sext(b, 32)
        if f3 == 0:
            return a - b if alt else a + b
        if f3 == 1:
            return a << (b & 31)
        if f3 == 2:
            return int(sa < sb)
        if f3 == 3:
            return int(a < b)
        if f3 == 4:
            return a ^ b
        if f3 == 5:
            return (sa >> (b & 31)) if alt else (a >> (b & 31))
        if f3 == 6:
            return a | b
        return a & b

    @staticmethod
    def muldiv(f3: int, a: int, b: int) -> int:
        sa, sb = _sext(a, 32), _sext(b, 32)
        if f3 == 0:
            return a * b
        if f3 == 1:
            return (sa * sb) >> 32
        if f3 == 2:
            return (sa * b) >> 32
        if f3 == 3:
            return (a * b) >> 32
        if f3 == 4:
            return -1 if sb == 0 else int(sa / sb) if sa != -(1 << 31) or sb != -1 else sa
        if f3 == 5:
            return 0xFFFFFFFF if b == 0 else a // b
        if f3 == 6:
            return sa if sb == 0 else 0 if (sa == -(1 << 31) and sb == -1) else sa - sb * int(sa / sb)
        return a if b == 0 else a % b

    def exec32(self, ins: int) -> None:
        x, pc = self.x, self.pc
        op, rd, f3 = ins & 0x7F, (ins >> 7) & 31, (ins >> 12) & 7
        rs1, rs2, f7 = (ins >> 15) & 31, (ins >> 20) & 31, ins >> 25
        imm_i = _sext(ins >> 20, 12)
        nxt = pc + 4
        if op == 0x37:
            self.set(rd, ins & 0xFFFFF000)
        elif op == 0x17:
            self.set(rd, pc + (ins & 0xFFFFF000))
        elif op == 0x6F:
            imm = ((ins >> 11) & 0x100000) | (ins & 0xFF000) | ((ins >> 9) & 0x800) | ((ins >> 20) & 0x7FE)
            self.set(rd, nxt)
            nxt = pc + _sext(imm, 21)
        elif op == 0x67:
            target = (x[rs1] + imm_i) & ~1
            self.set(rd, nxt)
            nxt = target
        elif op == 0x63:
            imm = ((ins >> 19) & 0x1000) | ((ins << 4) & 0x800) | ((ins >> 20) & 0x7E0) | ((ins >> 7) & 0x1E)
            a, b = x[rs1], x[rs2]
            take = {0: a == b, 1: a != b, 4: _sext(a, 32) < _sext(b, 32),
                    5: _sext(a, 32) >= _sext(b, 32), 6: a < b, 7: a >= b}[f3]
            if take:
                nxt = pc + _sext(imm, 13)
        elif op == 0x03:
            size, signed = {0: (1, True), 1: (2, True), 2: (4, False), 4: (1, False), 5: (2, False)}[f3]
            self.set(rd, self.load((x[rs1] + imm_i) & 0xFFFFFFFF, size, signed))
        elif op == 0x23:
            imm = _sext(((ins >> 20) & 0xFE0) | ((ins >> 7) & 31), 12)
            self.store((x[rs1] + imm) & 0xFFFFFFFF, 1 << f3, x[rs2])
        elif op == 0x13:
            shift = f3 in (1, 5)
            b = rs2 if shift else imm_i & 0xFFFFFFFF
            self.set(rd, self.alu(f3, x[rs1], b, shift and f7 == 0x20))
        elif op == 0x33:
            if f7 == 1:
                self.set(rd, self.muldiv(f3, x[rs1], x[rs2]))
            else:
                self.set(rd, self.alu(f3, x[rs1], x[rs2], f7 == 0x20))
        elif op == 0x73 and f3 in (1, 2, 3, 5, 6, 7):
            csr = ins >> 20
            old = self.csr.get(csr, 0)
            src = x[rs1] if f3 < 4 else rs1
            self.csr[csr] = src if f3 & 3 == 1 else old | src if f3 & 3 == 2 else old & ~src
            self.set(rd, old)
        elif op == 0x0F:
            pass
        else:
            raise AssertionError(f'unsupported instruction {ins:#010x} at {pc:#x}')
        self.pc = nxt & 0xFFFFFFFF

    def exec16(self, ins: int) -> None:
        x, pc = self.x, self.pc
        q, f3 = ins & 3, ins >> 13
        r = lambda v: 8 + (v & 7)          # rd', rs1', rs2'
        rd = (ins >> 7) & 31
        rs2 = (ins >> 2) & 31
        imm6 = _sext(((ins >> 7) & 0x20) | ((ins >> 2) & 31), 6)
        nxt = pc + 2
        if q == 0:
            off = ((ins >> 7) & 0x38) | ((ins >> 4) & 4) | ((ins << 1) & 0x40)
            if f3 == 0:
                imm = ((ins >> 7) & 0x30) | ((ins >> 1) & 0x3C0) | ((ins >> 4) & 4) | ((ins >> 2) & 8)
                self.set(r(ins >> 2), x[2] + imm)
            elif f3 == 2:
                self.set(r(ins >> 2), self.load((x[r(ins >> 7)] + off) & 0xFFFFFFFF, 4, False))
            elif f3 == 6:
                self.store((x[r(ins >> 7)] + off) & 0xFFFFFFFF, 4, x[r(ins >> 2)])
            else:
                raise AssertionError(f'unsupported instruction {ins:#06x} at {pc:#x}')
        elif q == 1:
            if f3 == 0:
                self.set(rd, x[rd] + imm6)
            elif f3 in (1, 5):
                imm = (((ins >> 1) & 0x800) | ((ins >> 7) & 0x10) | ((ins >> 1) & 0x300) | ((ins << 2) & 0x400)
                       | ((ins >> 1) & 0x40) | ((ins << 1) & 0x80) | ((ins >> 2) & 0xE) | ((ins << 3) & 0x20))
                if f3 == 1:
                    self.set(1, nxt)
                nxt = pc + _sext(imm, 12)
            elif f3 == 2:
                self.set(rd, imm6)
            elif f3 == 3:
                if rd == 2:
                    imm = (((ins >> 3) & 0x200) | ((ins >> 2) & 0x10) | ((ins << 1) & 0x40)
                           | ((ins << 4) & 0x180) | ((ins << 3) & 0x20))
                    self.set(2, x[2] + _sext(imm, 10))
                else:
                    self.set(rd, _sext(((ins << 5) & 0x20000) | ((ins << 10) & 0x1F000), 18))
            elif f3 == 4:
                d = r(ins >> 7)
                f2 = (ins >> 10) & 3
                shamt = ((ins >> 7) & 0x20) | ((ins >> 2) & 31)
                if f2 == 0:
                    self.set(d, x[d] >> shamt)
                elif f2 == 1:
                    self.set(d, _sext(x[d], 32) >> shamt)
                elif f2 == 2:
                    self.set(d, x[d] & (imm6 & 0xFFFFFFFF))
                else:
                    s = x[r(ins >> 2)]
                    self.set(d, [x[d] - s, x[d] ^ s, x[d] | s, x[d] & s][(ins >> 5) & 3])
            else:
                imm = (((ins >> 4) & 0x100) | ((ins >> 7) & 0x18) | ((ins << 1) & 0xC0)
                       | ((ins >> 2) & 6) | ((ins << 3) & 0x20))
                if (x[r(ins >> 7)] == 0) == (f3 == 6):
                    nxt = pc + _sext(imm, 9)
        else:
            if f3 == 0:
                self.set(rd, x[rd] << (((ins >> 7) & 0x20) | rs2))
            elif f3 == 2:
                off = ((ins >> 7) & 0x20) | ((ins >> 2) & 0x1C) | ((ins << 4) & 0xC0)
                self.set(rd, self.load((x[2] + off) & 0xFFFFFFFF, 4, False))
            elif f3 == 4:
                if not (ins >> 12) & 1:
                    if rs2 == 0:
                        nxt = x[rd] & ~1
                    else:
                        self.set(rd, x[rs2])
                elif rs2 == 0:
                    if rd == 0:
                        raise AssertionError(f'c.ebreak at {pc:#x}')
                    target = x[rd] & ~1
                    self.set(1, nxt)
                    nxt = target
                else:
                    self.set(rd, x[rd] + x[rs2])
            elif f3 == 6:
                off = ((ins >> 7) & 0x3C) | ((ins >> 1) & 0xC0)
                self.store((x[2] + off) & 0xFFFFFFFF, 4, x[rs2])
            else:
                raise AssertionError(f'unsupported instruction {ins:#06x} at {pc:#x}')
        self.pc = nxt & 0xFFFFFFFF


# ------------------------------------------------------------------ tests

def write_model(d: Path, device: str, m: int, k: int, n: int, seed: int = 0) -> Path:
    """What pynqo-middle-end writes for one INT8 GEMM assigned to ``device``.

    ``d/model.mlir`` is the CPU graph: a call to the bodiless declaration
    ``@__delegate_0``. ``d/delegates/<device>.mlir`` is the device module: the
    subgraph ``@<device>_0`` (the weight, a zero accumulator and the matmul).
    The weight is the same for every device, so the packages can be compared.
    """
    w = np.random.default_rng(seed).integers(-128, 128, size=(k, n), dtype=np.int8)
    rows = ', '.join('[' + ', '.join(str(int(v)) for v in row) + ']' for row in w)
    x, y = f'tensor<{m}x{k}xi8>', f'tensor<{m}x{n}xi32>'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'model.mlir').write_text(f'''module attributes {{pynqo.devices = ["cpu", "{device}"]}} {{
  func.func @forward(%x: {x}) -> {y} {{
    %r = call @__delegate_0(%x) : ({x}) -> {y}
    return %r : {y}
  }}
  func.func private @__delegate_0({x}) -> {y} attributes {{pynqo.delegate_id = 0 : i64, pynqo.device = "{device}"}}
}}
''', encoding='utf-8')
    (d / 'delegates').mkdir(exist_ok=True)
    (d / 'delegates' / f'{device}.mlir').write_text(f'''module @{device} attributes {{pynqo.device = "{device}"}} {{
  func.func @{device}_0(%arg0: {x}) -> {y} attributes {{pynqo.delegate_id = 0 : i64}} {{
    %w = arith.constant dense<[{rows}]> : tensor<{k}x{n}xi8>
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : {y}
    %z = linalg.fill ins(%c0 : i32) outs(%e : {y}) -> {y}
    %r = linalg.matmul ins(%arg0, %w : {x}, tensor<{k}x{n}xi8>) outs(%z : {y}) -> {y}
    return %r : {y}
  }}
}}
''', encoding='utf-8')
    return d / 'model.mlir'


def run_backend(model: Path, out: Path, devices: str, segment_bytes: int) -> None:
    pipeline._linux(f"{backend_tool()} '{pipeline._linux_path(model)}' --out '{pipeline._linux_path(out)}' "
                    f"--devices {devices} --cpus arm-cortex-a9 --segment-bytes {segment_bytes}")


def npu_jobs(package: Path) -> list[list[tuple]]:
    """Per segment, the jobs of the npu ISA program (ADDR_B inside the segment's block)."""
    words = isa.unpack((package / 'programs.bin').read_bytes())
    lines = (package / 'programs.txt').read_text().split('\n')
    segments = []
    for line in lines:
        f = line.split()
        if f and f[0] == 'seg':
            offset, length = int(f[1]), int(f[2])
            trace = isa_sim.run(words[offset:offset + length], np.zeros(1 << 24, np.uint8), log_jobs=True)
            segments.append(trace.jobs_log)
    return segments


def rv_jobs(package: Path, base=0x0010_0000, data_base=0x0080_0000) -> list[list[tuple]]:
    """Per segment, the jobs its binary issues, as offsets like npu_jobs."""
    blob = (package / 'rv_programs.bin').read_bytes()
    table = json.loads((package / 'rv_programs.json').read_text())
    segments = []
    for task in table['tasks']:
        for seg in task['segments']:
            image = blob[seg['bin_offset']:seg['bin_offset'] + seg['bin_bytes']]
            cpu = Rv32(image, seg['mem_bytes'], base, data_base)
            cpu.run()
            assert cpu.done == 0, f'DONE code {cpu.done}'
            b0 = data_base + task['b_base']
            segments.append([(m, n, k, a - data_base, b - b0, c - data_base)
                             for m, n, k, a, b, c in cpu.jobs])
    return segments


@unittest.skipUnless(tool_is_built(), 'pynqo-backend is not built')
class RiscvNpuBackendTests(unittest.TestCase):
    def check(self, m: int, k: int, n: int, segment_bytes: int, segments: int) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            run_backend(write_model(d / 'in-npu', 'npu', m, k, n), d / 'npu', 'cpu,npu', segment_bytes)
            run_backend(write_model(d / 'in-rv', 'risc_v_npu', m, k, n), d / 'rv', 'cpu,risc_v_npu',
                        segment_bytes)
            expected = npu_jobs(d / 'npu')
            got = rv_jobs(d / 'rv')
            self.assertEqual(len(got), segments)
            self.assertTrue(all(got), 'a segment issued no job')
            self.assertEqual(got, expected)
            # The delegate table: the subgraph's argument and result, M x K
            # INT8 in and M x N INT32 out, however many K chunks the device runs.
            for package, device in ((d / 'npu', 'npu'), (d / 'rv', 'risc_v_npu')):
                compiled = json.loads((package / 'compile.json').read_text())
                self.assertEqual(compiled['format'], 'pynqo-backend/2')
                self.assertEqual([(e['id'], e['device'], e['index'], e['input_bytes'], e['output_bytes'])
                                  for e in compiled['delegates']], [(0, device, 0, m * k, m * n * 4)])
            # The B tiles are the npu's: the same matrix-core layout.
            self.assertEqual((d / 'rv' / 'weights.bin').read_bytes(),
                             (d / 'npu' / 'weights.bin').read_bytes())
            table = (d / 'rv' / 'rv_programs.txt').read_text().split('\n')
            self.assertTrue(table[0].startswith('rv-programs 1 16 16 256 '))
            blob = (d / 'rv' / 'rv_programs.bin').read_bytes()
            self.assertEqual(len(blob) % 8, 0)
            self.assertEqual(int(table[0].split()[5]), len(blob))
            # The driver gets M, N, K and tk; it sums ceil(K / tk) planes.
            self.assertEqual(table[1].split()[:8], ['task', '0', str(m), str(n), str(k), str(min(m, 16)),
                                                    str(min(n, 16)), str(min(k, 256))])

    def test_one_segment_with_partial_tiles(self):
        self.check(33, 320, 20, 4 << 20, 1)

    def test_segments_and_a_narrow_last_tile(self):
        # One B panel per segment: three binaries, the last one narrow.
        self.check(17, 300, 40, 1, 3)

    def test_single_tile(self):
        self.check(4, 8, 8, 4 << 20, 1)

    def test_a_subgraph_that_is_not_one_matmul_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model = write_model(d / 'in', 'risc_v_npu', 4, 8, 8)
            module = d / 'in' / 'delegates' / 'risc_v_npu.mlir'
            # A non-zero accumulator: the delegate has only the activation as input.
            module.write_text(module.read_text().replace('arith.constant 0 : i32', 'arith.constant 1 : i32'))
            with self.assertRaisesRegex(RuntimeError, 'one claimable linalg.matmul'):
                run_backend(model, d / 'rv', 'cpu,risc_v_npu', 4 << 20)


if __name__ == '__main__':
    unittest.main()
