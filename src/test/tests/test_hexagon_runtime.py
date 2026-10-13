"""Host tests for the Hexagon subsystem driver (src/runtime/hexagon.py).

A fake MMIO emulates npu_hx_regs closely enough for the driver logic: the
identification registers, START -> running -> halted/fault after a few STATUS
reads (with the sticky events of STATUS bits 10/11), STOP -> halted with halt
cause 2, FLUSH busy, the TCM DMA copying between fake contiguous buffers and a
TCM array, the host TCM window and the mailboxes. The reference vector
kernels are checked on the ISS and the vector model. VectorOp is checked
against the fake with a program effect that applies the kernel to the fake
TCM, and its MMIO accesses per call are counted. No pynq is needed.
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
        self.stop_pc = 0x1234_5670   # where a STOP lands
        self.flush_reads = 0
        self.flush_reads_per_flush = 2
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
            self.flush_reads = self.flush_reads_per_flush
            self.flushes += 1
        if value & hx.CONTROL_START and not self.status & hx.STATUS_RUNNING:
            pc = self.rw[hx.REG_START_PC]
            self.starts.append(pc)
            self.status &= ~(hx.STATUS_HALTED | hx.STATUS_FAULT | hx.STATUS_HALT_EVENT
                             | hx.STATUS_FAULT_EVENT)
            self.status |= hx.STATUS_RUNNING
            self.cycles = self.instret = self.vec_cycles = 0
            self.pending = [self.program.reads, self._stop]
        if (value & hx.CONTROL_STOP and self.status & hx.STATUS_RUNNING
                and self.version >= hx.HX_VERSION_STOP):
            self.pending = None
            self.status &= ~hx.STATUS_RUNNING
            self.status |= hx.STATUS_HALTED | hx.STATUS_HALT_EVENT
            self.cause = hx.HALT_STOP << hx.CAUSE_HALT_SHIFT
            self.fault_pc = self.stop_pc
            self.cycles, self.instret = 999, 333
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


class CountingMmio:
    """Counts the driver's MMIO accesses on their way to a FakeHx."""

    def __init__(self, target):
        self.target = target
        self.reads = 0
        self.writes = 0

    def read(self, offset):
        self.reads += 1
        return self.target.read(offset)

    def write(self, offset, value):
        self.writes += 1
        self.target.write(offset, value)

    def take(self):
        counts = (self.writes, self.reads)
        self.reads = self.writes = 0
        return counts


def kernel_effect(kind):
    """What the reference kernel does to the TCM (MAILBOX0 = W, MAILBOX1 = a)."""
    from src.test.model import hx_vector as hv

    def effect(f):
        w, a_bits = f.rw[hx.REG_MAILBOX[0]], f.rw[hx.REG_MAILBOX[1]]
        av = hv.splat(hv.TYPE_FP16, a_bits, 64)
        for i in range(w):
            xw, yw = f.tcm[i], f.tcm[w + i]
            if kind == "int8":
                f.tcm[2 * w + i] = hv.vector_op(hv.OP_VADD, hv.TYPE_INT8, xw, yw, 0)
            else:
                prod = hv.vector_op(hv.OP_VMUL, hv.TYPE_FP16, av, xw, 0)
                f.tcm[2 * w + i] = hv.vector_op(hv.OP_VADD, hv.TYPE_FP16, prod, yw, 0)
    return effect


def make_minimal(kind="int8"):
    """A fake that finishes every operation on the first STATUS read (the
    board case: the hardware is faster than one MMIO access), counted."""
    alloc = FakeAllocator()
    fake = FakeHx(alloc)
    fake.dma_reads = 1
    fake.flush_reads_per_flush = 1
    fake.program = Program(reads=1, cycles=1702, instret=481, vec_cycles=1027,
                           effect=kernel_effect(kind))
    counter = CountingMmio(fake)
    npu = hx.HexagonNpu(mmio=counter, allocate=alloc, clock=FakeClock())
    counter.take()
    return npu, fake, alloc, counter


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

    def test_timeout_stops_the_core(self):
        npu, fake, _ = make()
        fake.program = Program(kind="hang")
        with self.assertRaises(hx.HexagonTimeout) as ctx:
            npu.run(0x1000, timeout=0.05)
        self.assertTrue(ctx.exception.status & hx.STATUS_RUNNING)   # as observed at the timeout
        self.assertIn("stopped at pc=0x12345670", str(ctx.exception))
        res = ctx.exception.result
        self.assertTrue(res.stopped and res.halted and not res.fault)
        self.assertEqual((res.cause_name, res.fault_pc, res.instret), ("stop", 0x1234_5670, 333))
        self.assertEqual(fake.flushes, 2)             # before START and after the stop
        self.assertEqual(fake.writes.count((hx.REG_CONTROL, hx.CONTROL_STOP)), 1)
        self.assertFalse(npu.status() & hx.STATUS_RUNNING)
        npu.start(0x1000)                             # the core is usable again

    def test_timeout_without_stop(self):
        npu, fake, _ = make()
        fake.program = Program(kind="hang")
        with self.assertRaises(hx.HexagonTimeout) as ctx:
            npu.run(0x1000, timeout=0.05, stop_on_timeout=False)
        self.assertTrue(ctx.exception.status & hx.STATUS_RUNNING)
        self.assertIsNone(ctx.exception.result)
        self.assertIn("running", str(ctx.exception))
        self.assertNotIn((hx.REG_CONTROL, hx.CONTROL_STOP), fake.writes)
        with self.assertRaisesRegex(hx.HexagonError, "still running"):
            npu.start(0x1000)

    def test_stop_and_resume(self):
        npu, fake, _ = make()
        self.assertTrue(npu.has_stop)
        fake.program = Program(kind="hang")
        npu.start(0x1000)
        res = npu.stop()
        self.assertTrue(res.stopped)
        self.assertEqual(res.halt_cause, hx.HALT_STOP)
        self.assertEqual(res.as_dict()["cause_name"], "stop")
        self.assertTrue(npu.status() & hx.STATUS_HALT_EVENT)
        # resume where it stopped; this time the program ends with ECALL
        fake.program = Program(instret=5, stop_pc=0x2000)
        npu.start(res.fault_pc)
        self.assertEqual(fake.starts, [0x1000, 0x1234_5670])
        done = npu.wait()
        self.assertFalse(done.stopped)
        self.assertEqual(done.cause_name, "ecall")

    def test_stop_idle_core(self):
        npu, fake, _ = make()
        npu.run(0x1000)
        res = npu.stop()                              # nothing to stop: last result
        self.assertTrue(res.halted)
        self.assertFalse(res.stopped)
        self.assertEqual(res.cause_name, "ecall")

    def test_stop_unsupported_version(self):
        npu, fake, _ = make(version=0x00010000)       # first build: no CONTROL.STOP
        self.assertFalse(npu.has_stop)
        with self.assertRaisesRegex(hx.HexagonError, "no CONTROL.STOP"):
            npu.stop()
        fake.program = Program(kind="hang")
        with self.assertRaises(hx.HexagonTimeout) as ctx:
            npu.run(0x1000, timeout=0.05)             # left running, as before
        self.assertIsNone(ctx.exception.result)
        self.assertNotIn((hx.REG_CONTROL, hx.CONTROL_STOP), fake.writes)

    def test_stop_timeout(self):
        npu, fake, _ = make()
        fake.program = Program(kind="hang")
        npu.start(0x1000)
        fake.version = 0                              # the fake ignores STOP now
        with self.assertRaisesRegex(hx.HexagonTimeout, "STOP"):
            npu.stop(timeout=0.02)

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
        self.assertEqual((hx.CONTROL_START, hx.CONTROL_FLUSH, hx.CONTROL_DMA_START, hx.CONTROL_CLEAR,
                          hx.CONTROL_STOP),
                         (g.C_START, g.C_FLUSH, g.C_DMA, g.C_CLEAR, g.C_STOP))
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
        # CONTROL bits and the CAUSE layout
        for bit, name in ((0, "core_start"), (1, "flush_start"), (4, "core_stop")):
            self.assertRegex(rtl, rf"{name} <= ctl_wr && acc_wdata\[{bit}\]")
        self.assertIn("R_CAUSE:    rd_word = {22'd0, halt_cause, 5'd0, fault_cause};", rtl)
        self.assertIn("input  logic [1:0]         halt_cause,", rtl)
        self.assertEqual(hx.CAUSE_HALT_MASK << hx.CAUSE_HALT_SHIFT, 0x300)


class VectorKernelTests(unittest.TestCase):
    """vector_kernel_source: assembles, and its vector commands (from the
    ISS, with the mailboxes readable) computed by the vector model match a
    word-by-word reference for every unroll."""

    def _run(self, kind, unroll, w, a_bits=0xBA00, seed=1):
        from src.test.model import hx_vector as hv
        from src.test.model.rv32_asm import assemble
        from src.test.model.rv32_iss import RV32ISS

        class MailboxISS(RV32ISS):
            def word(self, addr):
                if 0x8000_0050 <= addr < 0x8000_0060:
                    return [w, a_bits, 0, 0][(addr - 0x8000_0050) >> 2]
                return super().word(addr)

        prog = assemble(hx.vector_kernel_source(kind, unroll), base=0)
        iss = MailboxISS(mem_size=0x1000)
        iss.load_words(prog.image())
        res = iss.run(0)
        self.assertEqual((res.status, res.cause), ("halt", 0))
        rng = np.random.default_rng(seed)
        if kind == "int8":
            x = [int(v) for v in rng.integers(0, 2**63, w)]
            y = [int(v) for v in rng.integers(0, 2**63, w)]
        else:
            def f16():
                lanes = (rng.integers(0, 2, 4) << 15) | (rng.integers(8, 23, 4) << 10) | rng.integers(0, 1024, 4)
                return hv.join_lanes([int(v) for v in lanes], 16)
            x = [f16() for _ in range(w)]
            y = [f16() for _ in range(w)]
        model = hv.VectorUnitModel(vlen=64, nregs=8, tcm_words=1024)
        model.tcm[:2 * w] = x + y
        for c in iss.vcmds:
            model.execute(hv.VectorCommand(op=c.op, typ=c.vtype, vd=c.vd, vs1=c.vs1, vs2=c.vs2,
                                           addr=c.scalar % 1024, scalar=c.scalar))
        self.assertFalse(model.illegal)
        av = hv.splat(hv.TYPE_FP16, a_bits, 64)
        for i in range(w):
            if kind == "int8":
                ref = hv.vector_op(hv.OP_VADD, hv.TYPE_INT8, x[i], y[i], 0)
            else:
                ref = hv.vector_op(hv.OP_VADD, hv.TYPE_FP16,
                                   hv.vector_op(hv.OP_VMUL, hv.TYPE_FP16, av, x[i], 0), y[i], 0)
            self.assertEqual(model.tcm[2 * w + i], ref, (kind, unroll, i))
        self.assertEqual(model.tcm[:2 * w], x + y)            # inputs untouched
        self.assertTrue(all(v == 0 for v in model.tcm[3 * w:]))
        return res, len(prog.words)

    def test_kernels_match_the_reference(self):
        for kind in hx.VECTOR_KERNELS:
            for unroll in (1, 2, 4):
                for w in (unroll, 4 * unroll, 12):
                    self._run(kind, unroll, w)

    def test_instruction_count_per_word(self):
        # scalar-bound: instructions per word = commands + 3 pointer bumps + 1 / unroll
        for kind, cmds in (("int8", 4), ("fp16", 5)):
            small, _ = self._run(kind, 4, 8)
            large, words = self._run(kind, 4, 64)
            self.assertEqual(large.instret - small.instret, (64 - 8) * (cmds + 3) + (64 - 8) // 4)
            self.assertLessEqual(4 * words, 0x100)            # the self-test's program slot

    def test_arguments(self):
        with self.assertRaises(ValueError):
            hx.vector_kernel_source("int16")
        with self.assertRaises(ValueError):
            hx.vector_kernel_source("int8", 3)


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


def int8_reference(x, y):
    return np.clip(x.astype(np.int16) + y.astype(np.int16), -128, 127).astype(np.int8)


def fp16_operands(rng, n):
    bits = ((rng.integers(0, 2, n) << 15) | (rng.integers(8, 23, n) << 10)
            | rng.integers(0, 1024, n)).astype(np.uint16)
    return bits.view(np.float16)


def fp16_reference(a, x, y):
    """Product rounded to FP16, then the sum rounded (the kernel's two roundings)."""
    with np.errstate(all="ignore"):
        prod = (np.float32(np.float16(a)) * x.astype(np.float32)).astype(np.float16)
        return (prod.astype(np.float64) + y.astype(np.float64)).astype(np.float16)


class VectorOpTests(unittest.TestCase):
    """VectorOp: results, the full / steady-state paths and their MMIO cost."""

    def test_int8_results_over_repeated_calls(self):
        npu, fake, alloc, _ = make_minimal("int8")
        op = hx.VectorOp(npu, "int8", 512)
        rng = np.random.default_rng(3)
        for n in (512, 512, 256, 512):
            x = rng.integers(-128, 128, n).astype(np.int8)
            y = rng.integers(-128, 128, n).astype(np.int8)
            np.testing.assert_array_equal(op(x, y), int8_reference(x, y))
        self.assertEqual(op.calls, 4)
        self.assertEqual(op.last_cycles, 1702)
        self.assertEqual(len(fake.starts), 4)
        self.assertEqual(set(fake.starts), {op.program.phys_addr})
        self.assertEqual(fake.flushes, 1)              # only before the first call

    def test_fp16_results_and_out_argument(self):
        npu, fake, _, _ = make_minimal("fp16")
        op = hx.VectorOp(npu, "fp16", 256)
        rng = np.random.default_rng(4)
        x, y = fp16_operands(rng, 256), fp16_operands(rng, 256)
        for a in (-0.75, 1.5):
            got = op(x, y, a)
            np.testing.assert_array_equal(got.view(np.uint16), fp16_reference(a, x, y).view(np.uint16))
        out = np.zeros(256, np.float16)
        self.assertIs(op(x, y, -0.75, out=out), out)
        np.testing.assert_array_equal(out.view(np.uint16), fp16_reference(-0.75, x, y).view(np.uint16))
        self.assertEqual(fake.rw[hx.REG_MAILBOX[1]], 0xBA00)

    def test_same_result_as_the_generic_sequence(self):
        npu, fake, _, _ = make_minimal("int8")
        rng = np.random.default_rng(5)
        x = rng.integers(-128, 128, 256).astype(np.int8)
        y = rng.integers(-128, 128, 256).astype(np.int8)
        op = hx.VectorOp(npu, "int8", 256)
        fast = op(x, y)
        inp = npu.allocate(64, dtype=np.uint8)
        inp[:] = np.concatenate([x.view(np.uint8), y.view(np.uint8)])[:64]
        # generic: the whole x | y staged in one buffer, kernel run with both flushes
        staged = npu.allocate(512, dtype=np.uint8)
        staged[:] = np.concatenate([x.view(np.uint8), y.view(np.uint8)])
        out = npu.allocate(256, dtype=np.uint8)
        npu.mailbox[0], npu.mailbox[1] = 32, 0
        npu.dma_to_tcm(staged, 0)
        npu.run(op.program.phys_addr)
        npu.dma_from_tcm(out, 64)
        np.testing.assert_array_equal(np.asarray(out).view(np.int8), fast)

    def test_mmio_accesses_per_call(self):
        npu, fake, _, counter = make_minimal("int8")
        rng = np.random.default_rng(6)
        x = rng.integers(-128, 128, 512).astype(np.int8)
        y = rng.integers(-128, 128, 512).astype(np.int8)
        # the generic sequence of the self-test (mailboxes, DMA in, run, DMA out)
        staged = npu.allocate(1024, dtype=np.uint8)
        out = npu.allocate(512, dtype=np.uint8)
        prog = npu.load_program(hx.vector_kernel_words("int8"))
        counter.take()
        npu.mailbox[0], npu.mailbox[1] = 64, 0
        npu.dma_to_tcm(staged, 0)
        npu.run(prog.phys_addr)
        npu.dma_from_tcm(out, 128)
        # 14 writes; reads: 2 per DMA, 2 per FLUSH, 1 halt poll, 5 result
        # (15 before review round 3: the STATUS checks after a poll re-read)
        self.assertEqual(counter.take(), (14, 13))
        op = hx.VectorOp(npu, "int8", 512)
        counter.take()
        op(x, y)                                       # first call: full path
        self.assertEqual(counter.take(), (13, 6))
        flushes, invalidates = op._in.flushes, op._out.invalidates
        for _ in range(3):
            op(x, y)
            # DMA in (4 W + 1 poll), START (1 W + 1 poll), CYCLES, DMA out (4 W + 1 poll)
            self.assertEqual(counter.take(), (9, 4))
        self.assertEqual((op._in.flushes - flushes, op._out.invalidates - invalidates), (3, 3))
        op(x, y, stats=True)                           # + CAUSE, FAULT_PC, INSTRET, VEC_CYCLES
        self.assertEqual(counter.take(), (9, 8))
        self.assertEqual((op.last_result.instret, op.last_result.vec_cycles), (481, 1027))
        op(x[:256], y[:256])                           # new W: the two mailbox writes again
        self.assertEqual(counter.take(), (11, 4))

    def test_foreign_write_forces_the_full_path(self):
        npu, fake, _, counter = make_minimal("int8")
        op = hx.VectorOp(npu, "int8", 256)
        x = np.arange(256, dtype=np.int16).astype(np.int8)
        op(x, x)
        op(x, x)
        self.assertEqual(fake.flushes, 1)
        npu.mailbox[2] = 7                             # anything else written
        start_pc_writes = [v for o, v in fake.writes if o == hx.REG_START_PC]
        counter.take()
        np.testing.assert_array_equal(op(x, x), int8_reference(x, x))
        self.assertEqual(fake.flushes, 2)
        self.assertEqual(counter.take(), (13, 6))
        self.assertEqual(len([v for o, v in fake.writes if o == hx.REG_START_PC]),
                         len(start_pc_writes) + 1)
        npu.mailbox[0] = 99                            # a stale MAILBOX0 is rewritten
        np.testing.assert_array_equal(op(x, x), int8_reference(x, x))
        self.assertEqual(fake.rw[hx.REG_MAILBOX[0]], 32)

    def test_timeout_stops_and_next_call_recovers(self):
        npu, fake, _, _ = make_minimal("int8")
        op = hx.VectorOp(npu, "int8", 256, timeout=0.05)
        x = np.ones(256, np.int8)
        op(x, x)
        effect = fake.program.effect
        fake.program = Program(kind="hang")
        with self.assertRaises(hx.HexagonTimeout) as ctx:
            op(x, x)
        self.assertTrue(ctx.exception.result.stopped)
        self.assertEqual(fake.writes.count((hx.REG_CONTROL, hx.CONTROL_STOP)), 1)
        fake.program = Program(reads=1, effect=effect)
        np.testing.assert_array_equal(op(x, x), np.full(256, 2, np.int8))
        self.assertEqual(fake.flushes, 2)              # full path again after the timeout

    def test_fault_and_new_sticky_errors_raise(self):
        npu, fake, _, _ = make_minimal("int8")
        op = hx.VectorOp(npu, "int8", 256)
        x = np.ones(256, np.int8)
        fake.program = Program(kind="fault", cause=1, reads=1)
        with self.assertRaises(hx.HexagonFault):
            op(x, x)

        def illegal(f):
            f.status |= hx.STATUS_VEC_ILLEGAL
        fake.program = Program(reads=1, effect=illegal)
        with self.assertRaisesRegex(hx.HexagonError, "vec_illegal"):
            op(x, x)
        fake.program = Program(reads=1, effect=kernel_effect("int8"))
        op(x, x)                                       # already set before the call: not new

    def test_busy_subsystem_and_argument_checks(self):
        npu, fake, _, _ = make_minimal("int8")
        with self.assertRaisesRegex(hx.HexagonError, "TCM"):
            hx.VectorOp(npu, "int8", 8 * 344 * 4)
        with self.assertRaisesRegex(hx.HexagonError, "multiple of 32"):
            hx.VectorOp(npu, "int8", 100)
        with self.assertRaises(ValueError):
            hx.VectorOp(npu, "int16", 64)
        op = hx.VectorOp(npu, "int8", 256)
        with self.assertRaisesRegex(hx.HexagonError, "int8"):
            op(np.zeros(256, np.int16), np.zeros(256, np.int16))
        with self.assertRaisesRegex(hx.HexagonError, "exceed"):
            op(np.zeros(512, np.int8), np.zeros(512, np.int8))
        with self.assertRaisesRegex(hx.HexagonError, "multiple"):
            op(np.zeros(48, np.int8), np.zeros(48, np.int8))
        with self.assertRaisesRegex(hx.HexagonError, "out"):
            op(np.zeros(256, np.int8), np.zeros(256, np.int8), out=np.zeros(128, np.int8))
        fake.program = Program(kind="hang")
        npu.start(0x1000)
        with self.assertRaisesRegex(hx.HexagonError, "busy"):
            op(np.zeros(256, np.int8), np.zeros(256, np.int8))
        no_vec, _, _ = make(caps=0x0010007D)
        with self.assertRaisesRegex(hx.HexagonError, "vector unit"):
            hx.VectorOp(no_vec, "int8", 64)

    def test_kernel_words_are_assembled_once(self):
        a = hx.vector_kernel_words("fp16", 2)
        self.assertIs(a, hx.vector_kernel_words("fp16", 2))
        from src.test.model.rv32_asm import assemble
        self.assertEqual(list(a), assemble(hx.vector_kernel_source("fp16", 2), base=0).words)


class FastMmioTests(unittest.TestCase):
    """The direct MMIO.array binding used on the board."""

    class Ip:
        """A pynq DefaultIP stand-in whose read/write must not be used."""

        def __init__(self, words):
            class Mmio:
                pass
            self.mmio = Mmio()
            self.mmio.array = np.zeros(words, dtype=np.uint32)
            self.calls = 0

        def read(self, offset):
            self.calls += 1
            return int(self.mmio.array[offset >> 2])

        def write(self, offset, value):
            self.calls += 1
            self.mmio.array[offset >> 2] = value

    def _ip(self, words=hx.HX_WINDOW_BYTES // 4):
        ip = self.Ip(words)
        ip.mmio.array[hx.REG_MAGIC >> 2] = hx.HX_MAGIC
        ip.mmio.array[hx.REG_VERSION >> 2] = hx.HX_VERSION
        ip.mmio.array[hx.REG_CAPS >> 2] = CAPS_FULL
        return ip

    def test_array_is_used_directly(self):
        ip = self._ip()
        npu = hx.HexagonNpu(mmio=ip)
        self.assertEqual(npu.caps.raw, CAPS_FULL)
        npu.mailbox[1] = -2
        npu.tcm_write(1023, [0x1122334455667788])
        self.assertEqual(int(ip.mmio.array[hx.REG_MAILBOX[1] >> 2]), MASK32 - 1)
        self.assertEqual(npu.tcm_read(1023), [0x1122334455667788])
        self.assertEqual(int(ip.mmio.array[(hx.TCM_WINDOW + 8 * 1023 + 4) >> 2]), 0x11223344)
        self.assertEqual(npu.write_epoch, 3)
        self.assertEqual(ip.calls, 0)

    def test_fallbacks(self):
        ip = self._ip()
        hx.HexagonNpu(mmio=ip, fast_mmio=False).mailbox[0] = 1
        self.assertEqual(ip.calls, 4)                  # MAGIC, VERSION, CAPS, MAILBOX0
        small = self._ip(words=0x100)                  # does not cover the TCM window
        hx.HexagonNpu(mmio=small).mailbox[0] = 1
        self.assertEqual(small.calls, 4)


if __name__ == "__main__":
    unittest.main()
