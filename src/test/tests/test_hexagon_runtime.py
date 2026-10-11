"""Host tests for the Hexagon subsystem driver (src/runtime/hexagon.py).

A fake MMIO emulates npu_hx_regs closely enough for the driver logic: the
identification registers, START -> running -> halted/fault after a few STATUS
reads (with the sticky events of STATUS bits 10/11), FLUSH busy, the TCM DMA
copying between fake contiguous buffers and a TCM array, the host TCM window
and the mailboxes. No pynq is needed.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from src.runtime import hexagon as hx

REPO = Path(__file__).resolve().parents[3]
MASK32 = 0xFFFFFFFF
CAPS_FULL = 0x0010087F       # CONFIG 0 of tb_npu_hx_subsystem: every unit, 1024 TCM words


class FakeBuffer(np.ndarray):
    """A numpy array with the pynq buffer extras the driver uses."""

    def __new__(cls, shape, dtype, physical_address):
        obj = np.zeros(shape, dtype=dtype).view(cls)
        obj.physical_address = physical_address
        obj.flushes = 0
        obj.invalidates = 0
        return obj

    def __array_finalize__(self, obj):
        self.physical_address = getattr(obj, "physical_address", 0)
        self.flushes = getattr(obj, "flushes", 0)
        self.invalidates = getattr(obj, "invalidates", 0)

    def flush(self):
        self.flushes += 1

    def invalidate(self):
        self.invalidates += 1


class FakeAllocator:
    def __init__(self, base=0x1600_0000):
        self.next = base
        self.buffers = []

    def __call__(self, shape, dtype):
        buf = FakeBuffer(shape, dtype, self.next)
        self.next += (buf.nbytes + 0xFFF) & ~0xFFF
        self.buffers.append(buf)
        return buf

    def locate(self, phys, nbytes):
        for buf in self.buffers:
            base = buf.physical_address
            if base <= phys and phys + nbytes <= base + buf.nbytes:
                return buf, phys - base
        raise AssertionError(f"no fake buffer at 0x{phys:08x}")


class Program:
    """What the fake core does after START: stop after ``reads`` STATUS reads."""

    def __init__(self, kind="halt", cause=0, stop_pc=0, cycles=100, instret=40, reads=3,
                 vec_cycles=0, effect=None):
        self.kind, self.cause, self.stop_pc = kind, cause, stop_pc
        self.cycles, self.instret, self.reads = cycles, instret, reads
        self.vec_cycles, self.effect = vec_cycles, effect


class FakeHx:
    """Register-level model of npu_hx_regs + the units it reports on."""

    def __init__(self, allocator, caps=CAPS_FULL, magic=hx.HX_MAGIC, version=hx.HX_VERSION):
        self.allocator = allocator
        self.caps, self.magic, self.version = caps, magic, version
        self.tcm_words = ((caps >> 16) & 0xFFFF) * 64
        self.tcm = [0] * self.tcm_words
        self.rw = {hx.REG_START_PC: 0, hx.REG_DMA_DDR_ADDR: 0, hx.REG_DMA_TCM_ADDR: 0,
                   hx.REG_DMA_LEN: 0, hx.REG_IRQ_ENABLE: 0}
        self.rw.update({off: 0 for off in hx.REG_MAILBOX})
        self.status = 0
        self.cause = 0
        self.fault_pc = 0
        self.cycles = self.instret = self.vec_cycles = 0
        self.cache = dict.fromkeys(("l1i_hits", "l1i_misses", "l1d_hits", "l1d_misses",
                                    "l2_hits", "l2_misses"), 0)
        self.program = Program()
        self.pending = None          # (countdown, action)
        self.flush_reads = 0
        self.flushes = 0
        self.starts = []
        self.writes = []
        self.dma_reads = 2

    # ---------------------------------------------------------------- bus
    def read(self, offset):
        if offset >= hx.TCM_WINDOW:
            word, lane = (offset - hx.TCM_WINDOW) >> 3, (offset >> 2) & 1
            if word >= self.tcm_words:
                return 0
            return (self.tcm[word] >> (32 * lane)) & MASK32
        if offset == hx.REG_STATUS:
            self._tick()
            return self.status
        fixed = {hx.REG_MAGIC: self.magic, hx.REG_VERSION: self.version, hx.REG_CAPS: self.caps,
                 hx.REG_CONTROL: 0, hx.REG_CAUSE: self.cause, hx.REG_FAULT_PC: self.fault_pc,
                 hx.REG_CYCLES: self.cycles, hx.REG_INSTRET: self.instret,
                 hx.REG_VEC_CYCLES: self.vec_cycles,
                 hx.REG_L1I_HITS: self.cache["l1i_hits"], hx.REG_L1I_MISSES: self.cache["l1i_misses"],
                 hx.REG_L1D_HITS: self.cache["l1d_hits"], hx.REG_L1D_MISSES: self.cache["l1d_misses"],
                 hx.REG_L2_HITS: self.cache["l2_hits"], hx.REG_L2_MISSES: self.cache["l2_misses"]}
        if offset in fixed:
            return fixed[offset]
        return self.rw.get(offset, 0)

    def write(self, offset, value):
        assert isinstance(value, int) and 0 <= value <= MASK32, value
        self.writes.append((offset, value))
        if offset >= hx.TCM_WINDOW:
            word, lane = (offset - hx.TCM_WINDOW) >> 3, (offset >> 2) & 1
            if word < self.tcm_words:
                keep = self.tcm[word] & ~(MASK32 << (32 * lane))
                self.tcm[word] = keep | (value << (32 * lane))
            return
        if offset == hx.REG_CONTROL:
            self._control(value)
        elif offset == hx.REG_DMA_LEN:
            self.rw[offset] = value & 0x8000FFFF
        elif offset == hx.REG_IRQ_ENABLE:
            self.rw[offset] = value & 7
        elif offset in self.rw:
            self.rw[offset] = value

    # ---------------------------------------------------------------- behaviour
    def _control(self, value):
        if value & hx.CONTROL_CLEAR:
            self.status &= ~(hx.STATUS_DMA_DONE | hx.STATUS_BUS_ERROR | hx.STATUS_HALT_EVENT
                             | hx.STATUS_FAULT_EVENT | hx.STATUS_VEC_ILLEGAL)
        if value & hx.CONTROL_FLUSH and not self.status & hx.STATUS_FLUSH_BUSY:
            self.status |= hx.STATUS_FLUSH_BUSY
            self.flush_reads = 2
            self.flushes += 1
        if value & hx.CONTROL_START and not self.status & hx.STATUS_RUNNING:
            pc = self.rw[hx.REG_START_PC]
            self.starts.append(pc)
            self.status &= ~(hx.STATUS_HALTED | hx.STATUS_FAULT | hx.STATUS_HALT_EVENT
                             | hx.STATUS_FAULT_EVENT)
            self.status |= hx.STATUS_RUNNING
            self.cycles = self.instret = self.vec_cycles = 0
            self.pending = [self.program.reads, self._stop]
        if value & hx.CONTROL_DMA_START and not self.status & hx.STATUS_DMA_BUSY:
            self.status &= ~(hx.STATUS_DMA_DONE | hx.STATUS_DMA_ERROR)
            self.status |= hx.STATUS_DMA_BUSY
            self.dma_countdown = self.dma_reads

    def _stop(self):
        p = self.program
        self.status &= ~hx.STATUS_RUNNING
        self.cycles, self.instret, self.vec_cycles = p.cycles, p.instret, p.vec_cycles
        self.fault_pc = p.stop_pc
        self.cache["l1i_hits"] += 10
        self.cache["l1i_misses"] += 2
        self.cache["l1d_hits"] += 30
        self.cache["l1d_misses"] += 5
        if p.kind == "halt":
            self.status |= hx.STATUS_HALTED | hx.STATUS_HALT_EVENT
            self.cause = p.cause << 8
        else:
            self.status |= hx.STATUS_FAULT | hx.STATUS_FAULT_EVENT
            self.cause = p.cause
        if p.effect:
            p.effect(self)

    def _tick(self):
        if self.pending is not None and self.program.kind != "hang":
            self.pending[0] -= 1
            if self.pending[0] <= 0:
                action = self.pending[1]
                self.pending = None
                action()
        if self.status & hx.STATUS_FLUSH_BUSY:
            self.flush_reads -= 1
            if self.flush_reads <= 0:
                self.status &= ~hx.STATUS_FLUSH_BUSY
        if self.status & hx.STATUS_DMA_BUSY:
            self.dma_countdown -= 1
            if self.dma_countdown <= 0:
                self._dma_finish()

    def _dma_finish(self):
        ddr = self.rw[hx.REG_DMA_DDR_ADDR]
        tcm = self.rw[hx.REG_DMA_TCM_ADDR]
        length = self.rw[hx.REG_DMA_LEN]
        words, to_ddr = length & 0xFFFF, bool(length >> 31)
        self.status &= ~hx.STATUS_DMA_BUSY
        self.status |= hx.STATUS_DMA_DONE
        if ddr & 7 or tcm + words > self.tcm_words:
            self.status |= hx.STATUS_DMA_ERROR
            return
        try:
            buf, off = self.allocator.locate(ddr, 8 * words)
        except AssertionError:
            self.status |= hx.STATUS_DMA_ERROR | hx.STATUS_BUS_ERROR
            return
        raw = buf.view(np.uint8)
        if to_ddr:
            data = b"".join(w.to_bytes(8, "little") for w in self.tcm[tcm:tcm + words])
            raw[off:off + 8 * words] = np.frombuffer(data, dtype=np.uint8)
        else:
            chunk = raw[off:off + 8 * words].tobytes()
            for i in range(words):
                self.tcm[tcm + i] = int.from_bytes(chunk[8 * i:8 * i + 8], "little")


class FakeClock:
    def __init__(self, step=0.001):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def make(caps=CAPS_FULL, **kwargs):
    alloc = FakeAllocator()
    fake = FakeHx(alloc, caps=caps, **kwargs)
    npu = hx.HexagonNpu(mmio=fake, allocate=alloc, clock=FakeClock())
    return npu, fake, alloc


class IdentificationTests(unittest.TestCase):
    def test_no_pynq_import(self):
        code = ("import sys; import src.runtime.hexagon as h; "
                "h.HexagonNpu  # noqa\nprint('pynq' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.strip()
        self.assertEqual(out, "False")

    def test_caps_decode_full(self):
        caps = hx.Caps.decode(CAPS_FULL)
        self.assertTrue(caps.scalar and caps.vector and caps.l1i and caps.l1d and caps.l2
                        and caps.tcm and caps.dma)
        self.assertEqual(caps.vlen_bits, 64)
        self.assertEqual(caps.tcm_words, 1024)

    def test_caps_decode_reduced_configs(self):
        no_vec = hx.Caps.decode(0x0010007D)
        self.assertFalse(no_vec.vector)
        self.assertEqual(no_vec.vlen_bits, 0)
        self.assertFalse(hx.Caps.decode(0x0010086F).l2)
        self.assertFalse(hx.Caps.decode(0x0010083F).dma)

    def test_bad_magic_and_version(self):
        with self.assertRaisesRegex(hx.HexagonError, "MAGIC"):
            make(magic=0x12345678)
        with self.assertRaisesRegex(hx.HexagonError, "VERSION"):
            make(version=0x00020000)
        npu, _, _ = make(version=0x00010003)         # minor revisions are accepted
        self.assertEqual(npu.info()["caps"]["tcm_words"], 1024)

    def test_overlay_binding(self):
        class Overlay:
            pass

        overlay = Overlay()
        overlay.npu_hx_0 = FakeHx(FakeAllocator())
        npu = hx.HexagonNpu(overlay, allocate=FakeAllocator())
        self.assertIs(npu.mmio, overlay.npu_hx_0)
        with self.assertRaisesRegex(hx.HexagonError, "npu_hx_1"):
            hx.HexagonNpu(overlay, name="npu_hx_1")
        with self.assertRaises(hx.HexagonError):
            hx.HexagonNpu()


class RunTests(unittest.TestCase):
    def test_load_program_and_run(self):
        npu, fake, alloc = make()
        prog = npu.load_program([0x00000013, 0x00000073], offset=4)
        buf = alloc.buffers[0]
        self.assertEqual(prog.phys_addr, buf.physical_address + 16)
        self.assertEqual(list(np.asarray(buf)[4:6]), [0x13, 0x73])
        self.assertEqual(buf.flushes, 1)
        fake.program = Program(cycles=123, instret=41, stop_pc=prog.phys_addr + 4)
        before = npu.counters()
        result = npu.run(prog.phys_addr)
        self.assertEqual(fake.starts, [prog.phys_addr])
        self.assertEqual(fake.flushes, 2)             # before START and after the halt
        self.assertTrue(result.halted)
        self.assertFalse(result.fault)
        self.assertEqual(result.cause_name, "ecall")
        self.assertEqual((result.cycles, result.instret), (123, 41))
        self.assertAlmostEqual(result.cpi, 123 / 41)
        self.assertGreater(result.seconds, 0)
        delta = npu.counters().delta(before)
        self.assertEqual((delta.l1i_hits, delta.l1i_misses, delta.l1d_hits), (10, 2, 30))
        self.assertAlmostEqual(delta.hit_rates()["l1d"], 30 / 35)
        self.assertIsNone(delta.hit_rates()["l2"])

    def test_wait_ignores_stale_halt(self):
        npu, fake, _ = make()
        npu.run(0x1000)
        # The second program's halt event appears only after 5 STATUS reads;
        # wait() must keep polling until then (it keys on the sticky event).
        fake.program = Program(reads=5, instret=7)
        npu.start(0x2000)
        self.assertEqual(npu.wait().instret, 7)
        self.assertIsNone(fake.pending)

    def test_ebreak_halt(self):
        npu, fake, _ = make()
        fake.program = Program(cause=1, stop_pc=0x1008)
        result = npu.run(0x1000)
        self.assertEqual(result.halt_cause, 1)
        self.assertEqual(result.cause_name, "ebreak")
        self.assertEqual(result.as_dict()["fault_pc"], "0x00001008")

    def test_fault_raises_with_decoded_cause(self):
        npu, fake, _ = make()
        fake.program = Program(kind="fault", cause=6, stop_pc=0x1004, instret=1, cycles=9)
        with self.assertRaises(hx.HexagonFault) as ctx:
            npu.run(0x1000)
        res = ctx.exception.result
        self.assertEqual((res.fault_cause, res.fault_pc, res.instret), (6, 0x1004, 1))
        self.assertIn("data bus error", str(ctx.exception))
        self.assertIn("0x00001004", str(ctx.exception))
        self.assertEqual(fake.flushes, 2)             # still flushed after the fault

    def test_fault_without_raise(self):
        npu, fake, _ = make()
        fake.program = Program(kind="fault", cause=3, stop_pc=0x1010)
        res = npu.run(0x1000, raise_on_fault=False)
        self.assertTrue(res.fault)
        self.assertEqual(res.cause_name, "misaligned load")

    def test_timeout(self):
        npu, fake, _ = make()
        fake.program = Program(kind="hang")
        with self.assertRaises(hx.HexagonTimeout) as ctx:
            npu.run(0x1000, timeout=0.05)
        self.assertTrue(ctx.exception.status & hx.STATUS_RUNNING)
        self.assertIn("running", str(ctx.exception))
        with self.assertRaisesRegex(hx.HexagonError, "still running"):
            npu.start(0x1000)

    def test_start_argument_checks(self):
        npu, _, _ = make()
        with self.assertRaisesRegex(hx.HexagonError, "aligned"):
            npu.start(0x1002)
        with self.assertRaisesRegex(hx.HexagonError, "DDR window"):
            npu.start(0x4000_0000)

    def test_flush_timeout(self):
        npu, fake, _ = make()
        fake.flush_reads = 10 ** 9
        fake.status |= hx.STATUS_FLUSH_BUSY
        with self.assertRaisesRegex(hx.HexagonError, "already in progress"):
            npu.flush()
        fake.status &= ~hx.STATUS_FLUSH_BUSY
        orig = fake._control

        def stuck(value):
            orig(value)
            fake.flush_reads = 10 ** 9

        fake._control = stuck
        with self.assertRaises(hx.HexagonTimeout):
            npu.flush(timeout=0.02)

    def test_clear_and_irq_enable(self):
        npu, fake, _ = make()
        npu.run(0x1000)
        self.assertTrue(npu.status() & hx.STATUS_HALT_EVENT)
        npu.clear()
        self.assertFalse(npu.status() & hx.STATUS_HALT_EVENT)
        self.assertTrue(npu.status() & hx.STATUS_HALTED)
        npu.irq_enable(0xFF)
        self.assertEqual(fake.rw[hx.REG_IRQ_ENABLE], 7)


class DmaTests(unittest.TestCase):
    def test_round_trip(self):
        npu, fake, alloc = make()
        src = npu.allocate(16, dtype=np.uint64)
        src[:] = np.arange(16, dtype=np.uint64) * 0x0101010101010101
        dst = npu.allocate(16, dtype=np.uint64)
        npu.dma_to_tcm(src, tcm_word=100)
        self.assertEqual(src.flushes, 1)
        self.assertEqual(fake.tcm[100:116], [int(v) for v in src])
        self.assertEqual(fake.rw[hx.REG_DMA_LEN], 16)
        npu.dma_from_tcm(dst, tcm_word=100)
        self.assertEqual(fake.rw[hx.REG_DMA_LEN], 0x8000_0000 | 16)
        self.assertEqual(dst.invalidates, 1)
        np.testing.assert_array_equal(np.asarray(dst), np.asarray(src))

    def test_raw_dma_arguments(self):
        npu, fake, _ = make()
        with self.assertRaisesRegex(hx.HexagonError, "aligned"):
            npu.dma(0x1600_0004, 0, 4, to_ddr=False)
        with self.assertRaisesRegex(hx.HexagonError, "length"):
            npu.dma(0x1600_0000, 0, 0, to_ddr=False)
        with self.assertRaisesRegex(hx.HexagonError, "TCM range"):
            npu.dma(0x1600_0000, 1020, 8, to_ddr=False)
        self.assertNotIn(hx.REG_CONTROL, [off for off, _ in fake.writes])

    def test_dma_error_is_reported(self):
        npu, fake, _ = make()
        with self.assertRaisesRegex(hx.HexagonError, "DMA error"):
            npu.dma(0x2000_0000, 0, 4, to_ddr=False)   # no buffer there: SLVERR in the fake

    def test_dma_timeout(self):
        npu, fake, _ = make()
        npu.allocate(8, dtype=np.uint64)
        fake.dma_reads = 10 ** 9
        with self.assertRaisesRegex(hx.HexagonTimeout, "TCM DMA"):
            npu.dma(0x1600_0000, 0, 8, to_ddr=False, timeout=0.02)

    def test_no_dma_caps(self):
        npu, _, _ = make(caps=0x0010083F)
        with self.assertRaisesRegex(hx.HexagonError, "no TCM DMA"):
            npu.dma(0x1600_0000, 0, 1, to_ddr=False)


class RegisterMapConsistencyTests(unittest.TestCase):
    """The driver's register map against the constants of the subsystem test
    generator, which tb_npu_hx_subsystem checks against the RTL (every
    offset, STATUS / CONTROL / IRQ_ENABLE bit and CAPS value below is
    exercised there), and against the RTL header itself."""

    def test_offsets_and_bits_match_the_simulated_map(self):
        from src.test.vectors import generate_hx_subsystem_vectors as g

        pairs = {
            hx.REG_MAGIC: g.MAGIC, hx.REG_VERSION: g.VERSION, hx.REG_CAPS: g.CAPS,
            hx.REG_CONTROL: g.CONTROL, hx.REG_STATUS: g.STATUS, hx.REG_START_PC: g.START_PC,
            hx.REG_CAUSE: g.CAUSE, hx.REG_FAULT_PC: g.FAULT_PC, hx.REG_CYCLES: g.CYCLES,
            hx.REG_INSTRET: g.INSTRET, hx.REG_L1I_HITS: g.L1I_HITS,
            hx.REG_L1I_MISSES: g.L1I_MISSES, hx.REG_L1D_HITS: g.L1D_HITS,
            hx.REG_L1D_MISSES: g.L1D_MISSES, hx.REG_L2_HITS: g.L2_HITS,
            hx.REG_L2_MISSES: g.L2_MISSES, hx.REG_DMA_DDR_ADDR: g.DMA_DDR,
            hx.REG_DMA_TCM_ADDR: g.DMA_TCM, hx.REG_DMA_LEN: g.DMA_LEN,
            hx.REG_IRQ_ENABLE: g.IRQ_EN, hx.REG_VEC_CYCLES: g.VEC_CYCLES,
            hx.TCM_WINDOW: g.TCM_WIN,
        }
        for driver, sim in pairs.items():
            self.assertEqual(driver, sim)
        self.assertEqual(list(hx.REG_MAILBOX), g.MBOX)
        self.assertEqual(hx.REG_MAILBOX0, g.MBOX[0])
        self.assertEqual((hx.CONTROL_START, hx.CONTROL_FLUSH, hx.CONTROL_DMA_START, hx.CONTROL_CLEAR),
                         (g.C_START, g.C_FLUSH, g.C_DMA, g.C_CLEAR))
        self.assertEqual(
            (hx.STATUS_RUNNING, hx.STATUS_HALTED, hx.STATUS_FAULT, hx.STATUS_VEC_BUSY,
             hx.STATUS_DMA_BUSY, hx.STATUS_FLUSH_BUSY, hx.STATUS_VEC_ILLEGAL, hx.STATUS_DMA_ERROR,
             hx.STATUS_DMA_DONE, hx.STATUS_BUS_ERROR, hx.STATUS_HALT_EVENT, hx.STATUS_FAULT_EVENT),
            (g.S_RUNNING, g.S_HALTED, g.S_FAULT, g.S_VBUSY, g.S_DBUSY, g.S_FBUSY, g.S_VILL,
             g.S_DERR, g.S_DDONE, g.S_BERR, g.S_HEVT, g.S_FEVT))
        self.assertEqual(g.TCM_WORDS, hx.Caps.decode(CAPS_FULL).tcm_words)

    def test_caps_of_every_simulated_configuration(self):
        from src.test.vectors import generate_hx_subsystem_vectors as g

        for name, vec, l2, dma in g.CONFIGS:
            caps = hx.Caps.decode(g.caps_of(vec, l2, dma))
            self.assertEqual((caps.vector, caps.l2, caps.dma), (bool(vec), bool(l2), bool(dma)), name)
            self.assertTrue(caps.scalar and caps.l1i and caps.l1d and caps.tcm, name)
            self.assertEqual(caps.vlen_bits, 64 if vec else 0, name)
            self.assertEqual(caps.tcm_words, g.TCM_WORDS, name)
        self.assertEqual(g.caps_of(1, 1, 1), CAPS_FULL)

    def test_rtl_header_register_indices(self):
        """npu_hx_regs.sv localparams R_* (word index) -> driver offsets."""
        import re

        rtl = (REPO / "src/hw/rtl/npu_matrix/npu_hexagon/npu_hx_regs.sv").read_text()
        index = {m.group(1): int(m.group(2), 16) * 4
                 for m in re.finditer(r"localparam logic \[4:0\] R_(\w+)\s*=\s*5'h([0-9A-Fa-f]+);", rtl)}
        expect = {"MAGIC": hx.REG_MAGIC, "VERSION": hx.REG_VERSION, "CAPS": hx.REG_CAPS,
                  "CONTROL": hx.REG_CONTROL, "STATUS": hx.REG_STATUS, "START_PC": hx.REG_START_PC,
                  "CAUSE": hx.REG_CAUSE, "FAULT_PC": hx.REG_FAULT_PC, "CYCLES": hx.REG_CYCLES,
                  "INSTRET": hx.REG_INSTRET, "L1I_HIT": hx.REG_L1I_HITS,
                  "L1I_MISS": hx.REG_L1I_MISSES, "L1D_HIT": hx.REG_L1D_HITS,
                  "L1D_MISS": hx.REG_L1D_MISSES, "L2_HIT": hx.REG_L2_HITS,
                  "L2_MISS": hx.REG_L2_MISSES, "DMA_DDR": hx.REG_DMA_DDR_ADDR,
                  "DMA_TCM": hx.REG_DMA_TCM_ADDR, "DMA_LEN": hx.REG_DMA_LEN,
                  "IRQ_EN": hx.REG_IRQ_ENABLE, "MBOX0": hx.REG_MAILBOX[0],
                  "MBOX1": hx.REG_MAILBOX[1], "MBOX2": hx.REG_MAILBOX[2],
                  "MBOX3": hx.REG_MAILBOX[3], "VEC_CYC": hx.REG_VEC_CYCLES}
        self.assertEqual(index, expect)
        self.assertIn(f"HX_MAGIC = 32'h{hx.HX_MAGIC >> 16:04X}_{hx.HX_MAGIC & 0xFFFF:04X};", rtl)
        self.assertIn(f"HX_VERSION = 32'h{hx.HX_VERSION >> 16:04X}_{hx.HX_VERSION & 0xFFFF:04X};", rtl)


class WindowAndMailboxTests(unittest.TestCase):
    def test_tcm_window(self):
        npu, fake, _ = make()
        npu.tcm_write(5, [0x1122334455667788, 0xFFFFFFFF00000001])
        self.assertEqual(fake.tcm[5:7], [0x1122334455667788, 0xFFFFFFFF00000001])
        self.assertIn((hx.TCM_WINDOW + 40, 0x55667788), fake.writes)
        self.assertIn((hx.TCM_WINDOW + 44, 0x11223344), fake.writes)
        self.assertEqual(npu.tcm_read(5, 2), [0x1122334455667788, 0xFFFFFFFF00000001])
        npu.tcm_write(1023, [7])
        self.assertEqual(npu.tcm_read(1023), [7])
        with self.assertRaisesRegex(hx.HexagonError, "window"):
            npu.tcm_read(1023, 2)
        with self.assertRaisesRegex(hx.HexagonError, "window"):
            npu.tcm_write(-1, [0])

    def test_mailbox(self):
        npu, fake, _ = make()
        self.assertEqual(len(npu.mailbox), 4)
        npu.mailbox[0] = 0x600D
        npu.mailbox[3] = -1
        self.assertEqual(fake.rw[hx.REG_MAILBOX[3]], MASK32)
        self.assertEqual(npu.mailbox.read_all(), [0x600D, 0, 0, MASK32])
        with self.assertRaises(IndexError):
            npu.mailbox[4]

    def test_program_writes_mailbox(self):
        npu, fake, _ = make()

        def effect(f):
            f.rw[hx.REG_MAILBOX[2]] = f.rw[hx.REG_MAILBOX[0]] + f.rw[hx.REG_MAILBOX[1]]

        fake.program = Program(effect=effect)
        npu.mailbox[0], npu.mailbox[1] = 40, 2
        npu.run(0x1000)
        self.assertEqual(npu.mailbox[2], 42)


if __name__ == "__main__":
    unittest.main()
