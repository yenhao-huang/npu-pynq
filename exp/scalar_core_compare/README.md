# Scalar core A/B: CUSTOM versus CVA6

This experiment compares the two scalar core implementations of the NPU
(plan [2026-1112-003](../../docs/plans/2026-1112-003-cva6-scalar-core.md),
section 2) on the same workloads, in RTL simulation:

| Variant | `SCALAR_IMPL` | What runs | Program |
|---|---|---|---|
| A | `CUSTOM` | today's 64-bit ISA sequencer (`scalar_core/custom/`) | the ISA test programs |
| B | `CVA6` | CVA6 v5.3.0, single issue, `cv32a60x`-derived configuration (`scalar_core/cva6/`), driving the matrix core through the MMIO job registers | RV32I translations of the same programs |

Both run inside the full 16 x 16 `npu_accelerator` (MAX_K 256, 100 MHz
single clock domain) with the same matrix core, DMA and AXI memory model. The
numbers below were written by `run.py`; none were typed by hand.

## Method

- Workloads: the seven cases of `src/test/vectors/isa_frontend_16x16.txt`
  (`generate_isa_vectors.py`): five GEMMs tiled by `layout.encode`
  (REPEAT / INCR loops, up to 18 jobs), `illegal_opcode` (one job, then an
  illegal instruction) and `bad_shape` (N = 17 on a 16-wide array).
- B's programs come from `src/test/vectors/risc_v_npu/isa_to_rv.py`, which
  translates each ISA program instruction by instruction with the in-tree
  assembler `rv32asm.py` (no RISC-V toolchain): register-only code, REPEAT
  becomes a counted loop, each GEMM is a DOORBELL store followed by the
  post-increment stores of the next addresses, END is a FENCE read and a DONE
  store. `illegal_opcode` has no RISC-V counterpart; the translator emits
  FENCE and DONE = 0x10 (ERR_ILLEGAL_OPCODE) at that point, which gives the
  same observable result as A. `bad_shape` needs no special handling: the
  DOORBELL of an N = 17 job latches ERR_BAD_SHAPE (0x11) in `cva6_npu_mmio`.
  The data region is moved behind the (larger) RISC-V binary; the program and
  data are otherwise byte-identical.
- Bench: `src/hw/tb/npu_accelerator/tb_scalar_core_cva6.sv`, derived from
  `tb_scalar_core.sv` (same AXI4 memory with pseudo-random ready / valid gaps
  from the same LFSR, same protocol checks: at most 16-beat 8-byte INCR
  bursts that never cross a 128-byte line, WLAST placement, no START into a
  full matrix-core queue, legal shapes only). It is compiled once per variant
  with Verilator 5.050 (`make -C src/test sim-cva6` and `sim-cva6-custom`), so
  A and B share bench, simulator and scheduling.
- Correctness: after each program the whole 128 KiB memory model is compared
  word for word with the golden image of `src/test/model/isa_sim.py` (the C
  tiles start poisoned with 0xA5), and ISA_STATUS, ISA_ERROR and ISA_JOBS
  are checked. A golden file with one word flipped (tried once by hand) is
  reported as a mismatch, so the comparison is live.
- Performance: `isa_cycles` read from the CSR (0x60) after the program.
  The bench also timestamps ISA_CONTROL.START, every matrix-core START and the
  fall of `isa_busy`, and on B probes CVA6's DDR port and the MMIO slave.
- Program size: A = ISA words x 8 bytes; B = RISC-V binary bytes.
- Cross-check: A's cycles are also taken from the existing iverilog bench
  `tb_scalar_core` (`make -C src/test sim`). They differ from A under
  Verilator by up to 2 cycles because the two simulators schedule the
  bench's control-path tasks differently; the A/B ratios use the Verilator
  numbers of both sides.

## Results

<!-- results:begin (written by run.py) -->
Commit `f74ef0e` (with uncommitted changes under src/ or exp/), branch `npu/issue174-a`.

### Correctness and cycles

| Case | Jobs | A cycles | B cycles | B / A | B - A | A image | B image | A cycles (iverilog) |
|---|---:|---:|---:|---:|---:|---|---|---:|
| gemm_16x16x256 | 1 | 1518 | 1661 | 1.09 | 143 | PASS | PASS | 1520 |
| gemm_5x37x300 | 6 | 2449 | 2656 | 1.08 | 207 | PASS | PASS | 2447 |
| gemm_1x49x40 | 4 | 403 | 558 | 1.38 | 155 | PASS | PASS | 401 |
| gemm_33x20x320 | 18 | 6629 | 6866 | 1.04 | 237 | PASS | PASS | 6629 |
| gemm_3x3x7 | 1 | 48 | 214 | 4.46 | 166 | PASS | PASS | 48 |
| illegal_opcode | 1 | 19 | 262 | 13.79 | 243 | PASS | PASS | 19 |
| bad_shape | 0 | 6 | 156 | 26.00 | 150 | PASS | PASS | 6 |

### Where the cycles go

Clock cycles from ISA_CONTROL.START to the first matrix-core START (first job), between consecutive STARTs (min / avg / max), and from the last START until busy falls (tail).

| Case | A first job | B first job | A job gap | B job gap | A tail | B tail |
|---|---:|---:|---|---|---:|---:|
| gemm_16x16x256 | 16 | 163 | - | - | 1503 | 1499 |
| gemm_5x37x300 | 17 | 172 | 204 / 439 / 767 | 205 / 447 / 772 | 235 | 250 |
| gemm_1x49x40 | 17 | 166 | 13 / 78 / 132 | 24 / 78 / 120 | 152 | 159 |
| gemm_33x20x320 | 16 | 174 | 138 / 375 / 805 | 32 / 379 / 826 | 228 | 244 |
| gemm_3x3x7 | 16 | 166 | - | - | 33 | 49 |
| illegal_opcode | 19 | 164 | - | - | 79 | 99 |
| bad_shape | - | - | - | - | - | - |

CVA6 side only: cycles from START to CVA6's first DDR read (reset release and I$ invalidation), CVA6 DDR reads (I$ line fills of 16 bytes; the programs read and write no DDR data), MMIO writes and reads, and the average cycles from an MMIO AW handshake to its B handshake (DOORBELL is held while the job queue is full).

| Case | First fetch | I$ fills | MMIO writes | MMIO reads | MMIO write latency |
|---|---:|---:|---:|---:|---:|
| gemm_16x16x256 | 68 | 9 | 9 | 2 | 3.0 |
| gemm_5x37x300 | 68 | 21 | 31 | 2 | 44.1 |
| gemm_1x49x40 | 68 | 13 | 19 | 2 | 3.0 |
| gemm_33x20x320 | 68 | 77 | 106 | 2 | 39.5 |
| gemm_3x3x7 | 68 | 9 | 9 | 2 | 3.0 |
| illegal_opcode | 68 | 9 | 9 | 2 | 3.0 |
| bad_shape | 68 | 5 | 6 | 1 | 3.0 |

### Program size

| Case | A: ISA words | A bytes | B: RV32I instructions | B bytes | B / A |
|---|---:|---:|---:|---:|---:|
| gemm_16x16x256 | 7 | 56 | 33 | 132 | 2.36 |
| gemm_5x37x300 | 19 | 152 | 82 | 328 | 2.16 |
| gemm_1x49x40 | 10 | 80 | 45 | 180 | 2.25 |
| gemm_33x20x320 | 73 | 584 | 307 | 1228 | 2.10 |
| gemm_3x3x7 | 7 | 56 | 33 | 132 | 2.36 |
| illegal_opcode | 8 | 64 | 34 | 136 | 2.12 |
| bad_shape | 3 | 24 | 20 | 80 | 3.33 |
<!-- results:end -->

## Reading the numbers

- Correctness: all seven programs leave the golden memory image, the expected
  job count and the expected error code on both A and B. This is the first
  simulation of `scalar_core_cva6`: CVA6 boots from `prog_addr` at DDR
  address 0, its uncached non-idempotent MMIO accesses reach
  `cva6_npu_mmio` as single-beat transactions, I$ refills appear on `m_axi`
  as 2-beat 8-byte INCR bursts that pass the bench's protocol checks, and the
  stop / reset / drain sequence ends every program. No RTL change in
  `scalar_core/cva6/` was needed.
- Fixed start-up cost of B, about 150 cycles before the first job (A: 16 to
  19): about 68 cycles from START to CVA6's first instruction fetch, of which
  64 are the I$ invalidating its 64 sets after reset (`cva6_icache.sv` starts
  in `FLUSH` and walks every index); the rest is the cold I$ misses of the
  prologue (each a full round trip through the random-gap memory model) and
  the prologue's MMIO accesses, among them the blocking load of DATA_BASE.
  The split of those remaining cycles between misses and MMIO was not
  measured separately.
- Per job, B keeps up: the average gap between matrix-core STARTs is within
  about 2 % of A on the multi-job GEMMs, because the matrix core is the
  bottleneck and CVA6's DOORBELL store simply waits (MMIO write latency of
  about 40 cycles on average there, versus 3 when the queue is free).
- End of program: B's tail after the last START is up to 20 cycles longer
  than A's (the FENCE read must return to the core, then the DONE store and
  the STOP / DRAIN sequence of `scalar_core_cva6`); on `gemm_16x16x256`, where
  one long job dominates, it is a few cycles shorter.
- So B / A is close to 1 for long GEMMs (1.04 to 1.09) and large for tiny
  ones: `gemm_3x3x7` is 4.5 x A, almost all of it the start-up cost.
- `illegal_opcode` and `bad_shape` cycles are not comparable: A's
  `isa_cycles` stops when the decoder raises the error (it counts only while
  the program runs, not the drain), while B's counts until the program has
  ended and drained.
- Program size: the RV32I translations are 2.1 to 3.3 x the ISA programs.
  The translator emits uncompressed RV32I only; CVA6 is built with RVC, so a
  compressing assembler or compiler would shrink this, which was not tried.

## Not measured

- Area (LUT / FF / BRAM / DSP), Fmax / WNS at 100 MHz and power: there is no
  Yosys or Vivado on the machine that produced these numbers.
  `build_overlay.tcl --scalar-impl CVA6 --elaborate-only` exists but has not
  been run.
- B' (dual issue, `cv32a65x`-derived): not built, not run.
- The board: nothing here ran on the PYNQ-Z1. The memory model is not the
  Zynq HP port; absolute cycle counts there will differ, for both A and B.
- Code produced by the back end (`LowerToRiscv`) rather than by
  `isa_to_rv.py`, compressed (RVC) code, and programs that use the stack or
  DDR data (the D$ path is not exercised: the translated programs touch DDR
  only through instruction fetch).

## Reproduce

From the repository root, with the CVA6 submodule and its nested submodules
initialized (see `src/hw/rtl/npu_accelerator/scalar_core/cva6/README.md`):

```bash
python exp/scalar_core_compare/run.py                  # build, simulate, print the tables
python exp/scalar_core_compare/run.py --update-readme  # also rewrite the Results block above
```

It needs Python 3 with numpy, Verilator with `--timing`, a C++20 compiler
and GNU make on Linux. On the Windows development host the Verilator of
MSYS2 generates the C++ and WSL (Ubuntu, g++) compiles and runs it; the
details are in the comment above `sim-cva6` in `src/test/Makefile`. Building
and simulating both variants plus the iverilog cross-check took about 1.5
minutes there.
