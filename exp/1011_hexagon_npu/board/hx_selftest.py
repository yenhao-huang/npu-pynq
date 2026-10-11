"""Board acceptance of the Hexagon-style NPU subsystem on the PYNQ-Z1 (issue #136).

Runs on the --hexagon overlay (npu_hx_0 at 0x43C10000 beside the unchanged
tensor accelerator npu_accelerator_0) and checks, in order:

  a. identify   MAGIC / VERSION / CAPS, a mailbox and TCM-window round trip
  b. scalar     an RV32I program in DDR (sum, then signed bubble sort of 64
                words) against the reference ISS (src/test/model/rv32_iss.py):
                memory image and INSTRET must match; reports cycles, CPI and
                the L1I / L1D / L2 hit rates of the run
  c. vector     INT8 residual add with saturation and FP16 a*x+y over
                --elements lanes: DDR -> TCM DMA, a scalar-driven vector loop,
                TCM -> DDR DMA; bit-exact against hx_vector and numpy; reports
                cycles, elements/s at 100 MHz, DMA times and the wall time of
                the same op in numpy on the Cortex-A9
  d. tensor     exp/1007_sw_stack/board/isa_gemm.py (legacy tile + ISA GEMMs)
                in a child process, proving the tensor path on this overlay

Writes --out (JSON) after every step and exits non-zero on any mismatch or
error. ``--host-check`` runs only the host side (assembly, ISS, vector model
and numpy references agree) without pynq or a board.

    python3 hx_selftest.py --overlay overlay/artifacts/npu_matrix.bit --out results/hx_selftest.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.runtime import hexagon as hxr  # noqa: E402
from src.test.model import hx_numeric as hn  # noqa: E402
from src.test.model import hx_vector as hv  # noqa: E402
from src.test.model.rv32_asm import assemble  # noqa: E402
from src.test.model.rv32_iss import RV32ISS  # noqa: E402

CLOCK_HZ = hxr.CLOCK_HZ
SEED = 0x136
EXPECTED_CAPS = 0x0010087F     # every unit, VLEN 64, 1024 TCM words

# ------------------------------------------------------------------ (b) scalar
SORT_WORDS = 64
SCALAR_BUF_BYTES = 0x800       # program at 0, data at DATA_OFF, results at RESULT_OFF
DATA_OFF = 0x400
RESULT_OFF = DATA_OFF + 4 * SORT_WORDS   # sum, xor, min, max after the sort

SCALAR_SOURCE = f"""
    # Position independent: the data base comes from AUIPC, so the same words
    # run at offset 0 in the ISS and at the buffer's physical address on the board.
start:
    auipc s0, 0
    addi  s0, s0, {DATA_OFF}      # s0 = data base (start + DATA_OFF)
    # sum and xor of the {SORT_WORDS} input words
    mv    a0, s0
    li    t1, {SORT_WORDS}
    li    s2, 0
    li    s3, 0
sum:
    lw    t2, 0(a0)
    add   s2, s2, t2
    xor   s3, s3, t2
    addi  a0, a0, 4
    addi  t1, t1, -1
    bnez  t1, sum
    # signed bubble sort in place
    li    a1, {SORT_WORDS - 1}
outer:
    mv    a2, s0
    mv    a3, a1
inner:
    lw    t2, 0(a2)
    lw    t3, 4(a2)
    ble   t2, t3, noswap
    sw    t3, 0(a2)
    sw    t2, 4(a2)
noswap:
    addi  a2, a2, 4
    addi  a3, a3, -1
    bnez  a3, inner
    addi  a1, a1, -1
    bnez  a1, outer
    # results after the array: sum, xor, min (first), max (last)
    sw    s2, {4 * SORT_WORDS}(s0)
    sw    s3, {4 * SORT_WORDS + 4}(s0)
    lw    t2, 0(s0)
    sw    t2, {4 * SORT_WORDS + 8}(s0)
    lw    t2, {4 * SORT_WORDS - 4}(s0)
    sw    t2, {4 * SORT_WORDS + 12}(s0)
    ecall
"""

# ------------------------------------------------------------------ (c) vector
# Both kernels read their arguments from the mailboxes: MAILBOX0 = W, the
# number of 64-bit TCM words per operand; MAILBOX1 = the FP16 coefficient.
# TCM layout: x at word 0, r / y at word W, the result at word 2W.
INT8_SOURCE = """
    lui   s0, 0x80000         # npu_hx_regs through the scalar MMIO window
    lw    t0, 0x50(s0)        # MAILBOX0 = W
    li    a0, 0
    mv    a1, t0
    add   a2, t0, t0
loop:
    vld      v1, a0
    vld      v2, a1
    vadd.i8  v3, v1, v2       # saturating INT8 add
    vst      v3, a2
    addi  a0, a0, 1
    addi  a1, a1, 1
    addi  a2, a2, 1
    addi  t0, t0, -1
    bnez  t0, loop
    vfence
    ecall
"""

FP16_SOURCE = """
    lui   s0, 0x80000
    lw    t0, 0x50(s0)        # MAILBOX0 = W
    lw    t1, 0x54(s0)        # MAILBOX1 = a (binary16 bits)
    vsplat.f16 v4, t1
    li    a0, 0
    mv    a1, t0
    add   a2, t0, t0
loop:
    vld      v1, a0
    vld      v2, a1
    vmul.f16 v5, v4, v1       # a * x, rounded to FP16
    vadd.f16 v6, v5, v2       # + y, rounded again
    vst      v6, a2
    addi  a0, a0, 1
    addi  a1, a1, 1
    addi  a2, a2, 1
    addi  t0, t0, -1
    bnez  t0, loop
    vfence
    ecall
"""

FP16_A = 0xBA00                # -0.75
VEC_PROG_STRIDE = 0x100        # bytes between the two kernels in the program buffer
SENTINEL = 0xCC


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ------------------------------------------------------------------ references
def scalar_inputs(rng: np.random.Generator) -> np.ndarray:
    values = rng.integers(-(1 << 31), 1 << 31, SORT_WORDS, dtype=np.int64)
    values[:4] = [-(1 << 31), (1 << 31) - 1, 0, -1]
    rng.shuffle(values)
    return values.astype(np.int32)


def scalar_reference(program_words: list[int], data: np.ndarray):
    """Run the program in the ISS (flat memory at 0) and return (result, image)."""
    iss = RV32ISS(mem_size=SCALAR_BUF_BYTES)
    iss.load_words({4 * i: w for i, w in enumerate(program_words)})
    iss.mem[DATA_OFF:DATA_OFF + 4 * SORT_WORDS] = data.astype("<i4").tobytes()
    result = iss.run(0)
    if result.status != "halt" or result.cause != 0:
        raise AssertionError(f"ISS did not halt with ECALL: {result}")
    # Independent numpy cross-check of the ISS
    s = np.sort(data.astype(np.int64))
    expect = np.concatenate([s, [int(data.astype(np.int64).sum()) & 0xFFFFFFFF,
                                 int(np.bitwise_xor.reduce(data.view(np.uint32))), s[0], s[-1]]])
    got = np.frombuffer(bytes(iss.mem[DATA_OFF:RESULT_OFF + 16]), dtype="<u4")
    if not np.array_equal(got, (expect & 0xFFFFFFFF).astype(np.uint32)):
        raise AssertionError("ISS result disagrees with numpy")
    return result, bytes(iss.mem)


def int8_inputs(rng, n):
    def one():
        v = rng.integers(-128, 128, n).astype(np.int8)
        edge = rng.random(n) < 0.3
        v[edge] = rng.choice(np.array([127, 126, -128, -127, 100, -100], dtype=np.int8), edge.sum())
        return v
    return one(), one()


def fp16_inputs(rng, n):
    def one():
        sign = rng.integers(0, 2, n).astype(np.uint16) << 15
        exp = rng.integers(8, 23, n).astype(np.uint16) << 10
        bits = sign | exp | rng.integers(0, 1024, n).astype(np.uint16)
        special = np.array([0x7C00, 0xFC00, 0x7E00, 0x0000, 0x8000, 0x0001, 0x83FF, 0x7BFF],
                           dtype=np.uint16)
        pick = rng.random(n) < 0.03
        bits[pick] = rng.choice(special, pick.sum())
        return bits.view(np.float16)
    return one(), one()


def int8_numpy(x, r):
    return np.clip(x.astype(np.int16) + r.astype(np.int16), -128, 127).astype(np.int8)


def canonical_nan16(bits: np.ndarray) -> np.ndarray:
    bits = bits.copy()
    bits[((bits & 0x7C00) == 0x7C00) & ((bits & 0x03FF) != 0)] = hn.FP16_CANONICAL_NAN
    return bits


def fp16_numpy(a_bits, x, y):
    """Bit-exact numpy reference: the product is exact in float32 and rounded
    once to FP16, the sum is exact in float64 and rounded once; NaN results
    are the canonical 0x7E00 (the hardware's contract)."""
    a = np.array([a_bits], dtype=np.uint16).view(np.float16)[0]
    with np.errstate(all="ignore"):
        prod = (np.float32(a) * x.astype(np.float32)).astype(np.float16)
        out = (prod.astype(np.float64) + y.astype(np.float64)).astype(np.float16)
    return canonical_nan16(out.view(np.uint16)).view(np.float16)


def words64(arr: np.ndarray) -> list[int]:
    return [int(w) for w in np.frombuffer(arr.tobytes(), dtype="<u8")]


def vector_model(kind: str, x: np.ndarray, y: np.ndarray, a_bits: int = 0) -> np.ndarray:
    """hx_vector reference, word by word, as the kernel issues the commands."""
    out = []
    if kind == "int8":
        for xw, yw in zip(words64(x), words64(y)):
            out.append(hv.vector_op(hv.OP_VADD, hv.TYPE_INT8, xw, yw, 0))
        dtype = np.int8
    else:
        av = hv.splat(hv.TYPE_FP16, a_bits, 64)
        for xw, yw in zip(words64(x), words64(y)):
            prod = hv.vector_op(hv.OP_VMUL, hv.TYPE_FP16, av, xw, 0)
            out.append(hv.vector_op(hv.OP_VADD, hv.TYPE_FP16, prod, yw, 0))
        dtype = np.float16
    return np.frombuffer(np.array(out, dtype="<u8").tobytes(), dtype=dtype)


class MailboxISS(RV32ISS):
    """The reference ISS with MAILBOX0-3 readable at 0x8000_0050 (host check)."""

    def __init__(self, mailbox, **kw):
        super().__init__(**kw)
        self.mailbox = mailbox

    def word(self, addr):
        if 0x8000_0050 <= addr < 0x8000_0060:
            return self.mailbox[(addr - 0x8000_0050) >> 2] & 0xFFFFFFFF
        return super().word(addr)


def vector_program_on_iss(program_words, w, a_bits, x, y):
    """Execute a kernel's vector commands (from the ISS) on the vector model."""
    iss = MailboxISS([w, a_bits, 0, 0], mem_size=0x1000)
    iss.load_words({4 * i: v for i, v in enumerate(program_words)})
    res = iss.run(0)
    if res.status != "halt":
        raise AssertionError(f"kernel did not halt in the ISS: {res}")
    model = hv.VectorUnitModel(vlen=64, nregs=8, tcm_words=1024)
    model.tcm[0:2 * w] = words64(x) + words64(y)
    for c in iss.vcmds:
        model.execute(hv.VectorCommand(op=c.op, typ=c.vtype, vd=c.vd, vs1=c.vs1, vs2=c.vs2,
                                       addr=c.scalar % 1024, scalar=c.scalar))
    if model.illegal:
        raise AssertionError("kernel issued an illegal vector command")
    out = np.array(model.tcm[2 * w:3 * w], dtype="<u8").tobytes()
    return out, res


def check_elements(n: int, tcm_words: int = 1024) -> None:
    """Both kernels keep three operands of n lanes in the TCM (FP16: 4 lanes
    per 64-bit word, INT8: 8)."""
    if n <= 0 or n % 8:
        raise ValueError("--elements must be a positive multiple of 8")
    if 3 * (n // 4) > tcm_words:
        raise ValueError(f"{n} FP16 elements need {3 * (n // 4)} TCM words (> {tcm_words}); "
                         f"use at most {tcm_words // 3 * 4 // 8 * 8}")


def time_numpy(fn, repeat=50):
    fn()
    samples = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return float(np.median(samples)), float(min(samples))


# ------------------------------------------------------------------ the steps
class SelfTest:
    def __init__(self, args):
        self.args = args
        self.report = {"pass": True, "steps": {}}
        self.rng = np.random.default_rng(SEED)
        self.programs = {"scalar": assemble(SCALAR_SOURCE, base=0).words,
                         "int8": assemble(INT8_SOURCE, base=0).words,
                         "fp16": assemble(FP16_SOURCE, base=0).words}
        for name in ("int8", "fp16"):
            if 4 * len(self.programs[name]) > VEC_PROG_STRIDE:
                raise AssertionError(f"{name} kernel larger than its program slot")
        if 4 * len(self.programs["scalar"]) > DATA_OFF:
            raise AssertionError("scalar program overlaps its data")
        self.npu = None
        self.buffers = []

    def save(self):
        if self.args.out:
            out = Path(self.args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(jsonable(self.report), indent=2))

    def step(self, name, fn):
        print(f"== {name}", flush=True)
        t = time.monotonic()
        try:
            entry = fn()
            entry.setdefault("pass", True)
        except Exception as error:  # record and continue: later steps still report
            entry = {"pass": False, "error": repr(error), "traceback": traceback.format_exc()}
        entry["step_seconds"] = time.monotonic() - t
        self.report["steps"][name] = entry
        self.report["pass"] = bool(self.report["pass"] and entry["pass"])
        print(json.dumps(jsonable({k: v for k, v in entry.items() if k != "traceback"})), flush=True)
        if "traceback" in entry:
            print(entry["traceback"], flush=True)
        self.save()

    def alloc(self, n, dtype):
        buf = self.npu.allocate(n, dtype=dtype)
        self.buffers.append(buf)
        return buf

    def free(self):
        for buf in self.buffers:
            try:
                buf.freebuffer()
            except Exception:
                pass
        self.buffers = []

    # (a)
    def identify(self):
        npu = self.npu
        caps = npu.caps
        entry = npu.info()
        problems = []
        for unit in ("scalar", "vector", "l1i", "l1d", "l2", "tcm", "dma"):
            if not getattr(caps, unit):
                problems.append(f"CAPS lacks {unit}")
        if caps.vlen_bits != 64:
            problems.append(f"VLEN {caps.vlen_bits}, expected 64")
        if caps.raw != EXPECTED_CAPS:
            entry["note"] = f"CAPS 0x{caps.raw:08x} differs from the default 0x{EXPECTED_CAPS:08x}"
        status = npu.status()
        entry["status_at_start"] = hxr.describe_status(status)
        if status & hxr.STATUS_RUNNING:
            problems.append("scalar core already running")
        npu.clear()
        pattern = [0x13600001, 0xA5A5A5A5, 0x5A5A5A5A, 0xFFFFFFFF]
        for i, v in enumerate(pattern):
            npu.mailbox[i] = v
        if npu.mailbox.read_all() != pattern:
            problems.append(f"mailbox readback {npu.mailbox.read_all()}")
        last = caps.tcm_words - 1
        probe = [0x0123456789ABCDEF, 0xFEDCBA9876543210]
        npu.tcm_write(last - 1, probe)
        got = npu.tcm_read(last - 1, 2)
        if got != probe:
            problems.append(f"TCM window readback {[hex(v) for v in got]}")
        entry["checks"] = problems or "ok"
        entry["pass"] = not problems
        return entry

    # (b)
    def scalar(self):
        npu = self.npu
        words = self.programs["scalar"]
        data = scalar_inputs(self.rng)
        ref, image = scalar_reference(words, data)
        buf = self.alloc(SCALAR_BUF_BYTES // 4, np.uint32)
        raw = np.asarray(buf).view(np.uint8)
        raw[:] = 0
        raw[DATA_OFF:DATA_OFF + 4 * SORT_WORDS] = np.frombuffer(data.astype("<i4").tobytes(), np.uint8)
        prog = npu.load_program(words, buf)                  # flushes the whole buffer
        before = npu.counters()
        result = npu.run(prog.phys_addr, timeout=self.args.timeout)
        delta = npu.counters().delta(before)
        buf.invalidate()
        got = bytes(raw[DATA_OFF:RESULT_OFF + 16])
        want = image[DATA_OFF:RESULT_OFF + 16]
        mismatches = [i // 4 for i in range(0, len(want), 4) if got[i:i + 4] != want[i:i + 4]]
        code_intact = bytes(raw[:4 * len(words)]) == image[:4 * len(words)]
        ok = not mismatches and result.instret == ref.instret and code_intact
        return {
            "pass": ok,
            "program_words": len(words), "start_pc": f"0x{prog.phys_addr:08x}",
            "run": result.as_dict(),
            "iss_instret": ref.instret,
            "iss_ideal_cycles": ref.cycles,      # 1-cycle-latency memory, no misses
            "cycles": result.cycles, "instret": result.instret, "cpi": result.cpi,
            "us_at_100mhz": result.cycles / CLOCK_HZ * 1e6,
            "counters": delta.as_dict(),
            "mismatched_words": mismatches[:16], "mismatch_count": len(mismatches),
            "code_intact": code_intact,
            "sum": int(np.frombuffer(got[4 * SORT_WORDS:4 * SORT_WORDS + 4], "<u4")[0]),
        }

    # (c)
    def vector(self):
        npu = self.npu
        n = self.args.elements
        check_elements(n, npu.caps.tcm_words)
        prog_buf = self.alloc(2 * VEC_PROG_STRIDE // 4, np.uint32)
        np.asarray(prog_buf)[:] = 0
        pcs = {"int8": npu.load_program(self.programs["int8"], prog_buf, 0).phys_addr,
               "fp16": npu.load_program(self.programs["fp16"], prog_buf, VEC_PROG_STRIDE // 4).phys_addr}
        x8, r8 = int8_inputs(self.rng, n)
        xf, yf = fp16_inputs(self.rng, n)
        kernels = {
            "int8_residual_add": ("int8", x8, r8, 0, int8_numpy(x8, r8),
                                  lambda: np.clip(x8.astype(np.int16) + r8, -128, 127).astype(np.int8)),
            "fp16_axpy": ("fp16", xf, yf, FP16_A, fp16_numpy(FP16_A, xf, yf),
                          lambda: np.float16(-0.75) * xf + yf),
        }
        entry = {"elements": n, "fp16_a": f"0x{FP16_A:04x}", "pass": True}
        for name, (kind, x, y, a_bits, ref_np, np_op) in kernels.items():
            w = x.nbytes // 8                      # TCM words per operand
            ref_model = vector_model(kind, x, y, a_bits)
            model_vs_numpy = bytes(ref_model.tobytes()) == bytes(ref_np.tobytes())
            # DDR in: x | y | sentinel (so a result the kernel never stored shows)
            inp = self.alloc(3 * w * 8, np.uint8)
            raw_in = np.asarray(inp)
            raw_in[:w * 8] = np.frombuffer(x.tobytes(), np.uint8)
            raw_in[w * 8:2 * w * 8] = np.frombuffer(y.tobytes(), np.uint8)
            raw_in[2 * w * 8:] = SENTINEL
            out = self.alloc(w * 8, np.uint8)
            np.asarray(out)[:] = SENTINEL
            out.flush()
            npu.mailbox[0] = w
            npu.mailbox[1] = a_bits
            t0 = time.perf_counter()
            dma_in = npu.dma_to_tcm(inp, tcm_word=0)
            run = npu.run(pcs[kind], timeout=self.args.timeout)
            dma_out = npu.dma_from_tcm(out, tcm_word=2 * w)
            wall = time.perf_counter() - t0
            got = np.asarray(out).tobytes()
            status = npu.status()
            diff = np.flatnonzero(np.frombuffer(got, np.uint8) != np.frombuffer(ref_model.tobytes(), np.uint8))
            ok = (not diff.size and model_vs_numpy and not status & hxr.STATUS_VEC_ILLEGAL)
            np_median, np_min = time_numpy(np_op)
            seconds_hw = run.cycles / CLOCK_HZ
            entry[name] = {
                "pass": ok, "tcm_words_per_operand": w,
                "model_matches_numpy": model_vs_numpy,
                "mismatched_bytes": int(diff.size), "first_mismatch_byte": int(diff[0]) if diff.size else None,
                "vec_illegal": bool(status & hxr.STATUS_VEC_ILLEGAL),
                "cycles": run.cycles, "vec_cycles": run.vec_cycles, "instret": run.instret,
                "cpi": run.cpi, "cycles_per_element": run.cycles / n,
                "elements_per_s_at_100mhz": n / seconds_hw if seconds_hw else None,
                "dma_in_seconds": dma_in, "dma_out_seconds": dma_out, "run_seconds": run.seconds,
                "end_to_end_seconds": wall,
                "numpy_a9_median_seconds": np_median, "numpy_a9_min_seconds": np_min,
                "numpy_a9_elements_per_s": n / np_median if np_median else None,
                "speedup_kernel_vs_numpy": np_median / seconds_hw if seconds_hw else None,
            }
            entry["pass"] = entry["pass"] and ok
        return entry

    # (d)
    def tensor(self):
        script = REPO / "exp" / "1007_sw_stack" / "board" / "isa_gemm.py"
        out = Path(self.args.tensor_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.monotonic()
        proc = subprocess.run([sys.executable, "-B", str(script), "--overlay", self.args.overlay,
                               "--out", str(out)], cwd=str(REPO), capture_output=True, text=True,
                              timeout=self.args.tensor_timeout)
        entry = {"returncode": proc.returncode, "seconds": time.monotonic() - t,
                 "report": str(out), "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
        if out.exists():
            gemm = json.loads(out.read_text())
            entry["version"] = gemm.get("version")
            entry["capabilities"] = gemm.get("capabilities")
            entry["legacy_tile"] = gemm.get("legacy_tile", {}).get("match")
            entry["legacy_after_isa"] = gemm.get("legacy_after_isa")
            entry["cases"] = [{k: c.get(k) for k in ("m", "n", "k", "match", "cycles", "gmacs")}
                              for c in gemm.get("cases", [])]
            entry["pass"] = proc.returncode == 0 and bool(gemm.get("pass"))
        else:
            entry["pass"] = False
        return entry

    # host-only consistency check
    def host_check(self):
        entry = {"pass": True}
        data = scalar_inputs(self.rng)
        ref, _ = scalar_reference(self.programs["scalar"], data)
        entry["scalar"] = {"program_words": len(self.programs["scalar"]), "iss_instret": ref.instret,
                           "iss_ideal_cycles": ref.cycles}
        n = self.args.elements
        check_elements(n)
        for kind, (x, y), a_bits in (("int8", int8_inputs(self.rng, n), 0),
                                     ("fp16", fp16_inputs(self.rng, n), FP16_A)):
            ref_np = int8_numpy(x, y) if kind == "int8" else fp16_numpy(a_bits, x, y)
            w = x.nbytes // 8
            model = vector_model(kind, x, y, a_bits)
            on_iss, res = vector_program_on_iss(self.programs[kind], w, a_bits, x, y)
            ok = model.tobytes() == ref_np.tobytes() == on_iss
            entry[kind] = {"pass": ok, "tcm_words_per_operand": w, "iss_instret": res.instret,
                           "vector_commands_per_word": 4 if kind == "int8" else 5}
            entry["pass"] = entry["pass"] and ok
        return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--overlay", default="overlay/artifacts/npu_matrix.bit")
    ap.add_argument("--out", default="results/hx_selftest.json")
    ap.add_argument("--name", default="npu_hx_0", help="IP name of the subsystem in the overlay")
    ap.add_argument("--elements", type=int, default=512, help="lanes per vector kernel")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per scalar program")
    ap.add_argument("--skip-tensor", action="store_true", help="skip the tensor ISA GEMM step")
    ap.add_argument("--tensor-out", default="results/isa_gemm.json")
    ap.add_argument("--tensor-timeout", type=float, default=1800.0)
    ap.add_argument("--host-check", action="store_true",
                    help="check programs and references on the host only (no pynq)")
    args = ap.parse_args(argv)

    test = SelfTest(args)
    if args.host_check:
        test.step("host_check", test.host_check)
        return 0 if test.report["pass"] else 1

    from pynq import Overlay

    t = time.monotonic()
    overlay = Overlay(args.overlay)
    test.report["overlay"] = {"path": args.overlay, "load_seconds": time.monotonic() - t,
                              "ips": sorted(overlay.ip_dict)}
    try:
        test.npu = hxr.HexagonNpu(overlay, args.name)
    except Exception as error:
        test.report["pass"] = False
        test.report["steps"]["identify"] = {"pass": False, "error": repr(error)}
        test.save()
        print(json.dumps(jsonable(test.report), indent=2))
        return 1
    test.step("identify", test.identify)
    test.step("scalar", test.scalar)
    test.step("vector", test.vector)
    test.free()
    if not args.skip_tensor:
        test.step("tensor", test.tensor)
    test.save()
    print("PASS" if test.report["pass"] else "FAIL", flush=True)
    return 0 if test.report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
