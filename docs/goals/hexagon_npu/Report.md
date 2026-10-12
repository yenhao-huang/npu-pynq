# Report: Hexagon-style heterogeneous NPU on the PYNQ-Z1 (#136)

Goal: [goal.md](goal.md). Architecture and interfaces:
[architecture.md](architecture.md). How to rerun everything:
[Reproduce.md](Reproduce.md). Branch `npu/issue136-a`, cut from
`npu/issue119-a` (the NPU ISA front end).

## What was built

Following the Qualcomm Hexagon NPU (Hot Chips 2023): scalar control, a
vector SIMD unit and a tensor unit sharing a TCM, with caches in front of
DDR. The existing modules keep their roles (DMA/BUS = `dma_axi` + AXI DMA +
the ISA front end's AXI master, TCM = the matrix operand buffers, tensor
unit = the 16 x 16 INT8 systolic array); the new units form a second
overlay cell next to them.

```text
PS7 (Cortex-A9) --GP0--> npu_accelerator_0 (unchanged: ISA front end, 16 x 16 INT8 tensor array)
                \-GP0--> npu_hx_0 = npu_hx_subsystem (new)
                           scalar core (RV32I + vector dispatch)
                             |- L1 I$ 1 KB --\
                             |- L1 D$ 1 KB ---+- L2 8 KB 2-way -- line bridge --\
                             |- TCM window                                      +- AXI4 arbiter --HP0--> DDR
                             `- MMIO (regs, DMA, flush)       TCM DMA ----------/
                           TCM 8 KB (dual port) <-- vector unit (INT8/16/32, FP16)
multi-precision tensor PE/array (INT8 / INT16 / FP16): verified and measured stand-alone
```

| Requirement (goal.md) | Delivered |
| --- | --- |
| L1/L2 cache | `npu_cache` as L1 I$ (1 KB, read-only), L1 D$ (1 KB, write-back), L2 (8 KB, 2-way LRU); arbiter and AXI4 line bridge |
| Scalar unit | `npu_scalar_core`: multi-cycle RV32I, custom-0 vector dispatch, faults, STOP; Python assembler and ISS |
| Vector unit | `npu_vector_unit`: 8 x 64-bit registers, INT8/16/32 and FP16 lanes, load/store to TCM |
| FP16, INT16 data widths | FP16 in the vector unit (on the board); INT8/INT16/FP16 multi-precision tensor PE and array (`npu_pe_mp`, `npu_systolic_array_mp`); IEEE binary16/binary32 arithmetic units |
| Modular, not coupled | each unit is a separate module with a valid/ready or req/gnt boundary; `npu_hx_subsystem` is the only place they meet, and `HX_ENABLE_VECTOR/L2/DMA` remove units; the tensor accelerator and its runtime are untouched (a second block-design cell) |

## Acceptance

| Criterion | Result | Evidence |
| --- | --- | --- |
| Dedicated testbench per component, functional simulation passes | PASS: 15 new self-checking testbenches (plus the 15 existing), all pass; lint clean | `make -C src/test lint sim` |
| PPA run per component, compared with before | PASS: Vivado 2026.1 out-of-context (area, Fmax, vectorless power) for every unit and for the subsystem with and without each unit; routed overlay with and without `--hexagon` | [PPA/](PPA/) |
| Report.md, Reproduce.md | this file, [Reproduce.md](Reproduce.md) | |
| Review loop: 3 rounds per component | PASS: 6 components x 3 rounds (memory, scalar, numerics/tensor, vector, cache, subsystem) | [Review rounds](#review-rounds) |
| Deploy on the board, run SmolLM | PASS: `--hexagon` overlay on the PYNQ-Z1, self-test of every new unit passes, SmolLM2 answers "The capital of France is Paris..." | [board.md](board.md) |

## PPA summary

Vivado 2026.1, `xc7z020clg400-1`, 10 ns. Out-of-context per unit (added
area; nothing comparable existed before):

| Unit | LUT | FF | DSP | BRAM | Fmax (MHz) | Dynamic (W) | Page |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Scalar core | 950 | 392 | 0 | 0 | 144.6 | 0.009 | [scalar.md](PPA/scalar.md) |
| L1 I$ + L1 D$ + L2 + arbiter + bridge | 1,125 | 777 | 0 | 6 | 148.5-266.7 | 0.017 | [cache.md](PPA/cache.md) |
| TCM + TCM DMA + AXI4 arbiter | 471 | 459 | 0 | 2 | 166.5-210.8 | 0.018 | [memory.md](PPA/memory.md) |
| Vector unit (incl. 2 FP16 add + 2 FP16 mul) | 1,324 | 793 | 6 | 0 | 101.7 | 0.036 | [vector.md](PPA/vector.md) |
| Subsystem, all units | 4,186 | 2,774 | 6 | 8 | 104.0 | 0.053 | [subsystem.md](PPA/subsystem.md) |

Subsystem deltas (what each unit adds, out of context): vector unit
+1,343 LUT / 6 DSP / +24 mW; L2 +471 LUT / 5 BRAM; TCM DMA + arbiter
+429 LUT; scalar core + L1s + TCM + registers (the minimum configuration)
1,921 LUT. Details in [subsystem.md](PPA/subsystem.md).

Whole overlay, routed (before = base branch build, after = `--hexagon`):

| | LUT | FF | Slices | DSP | BRAM | WNS (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | 35,477 (66.7 %) | 25,200 | 11,141 (83.8 %) | 194 | 26.5 | +0.031 |
| After (final RTL, run 38205853103) | 40,011 (75.2 %) | 28,326 | 12,215 (91.8 %) | 200 | 34.5 | +0.113 |
| Delta | +4,534 | +3,126 | +1,074 | +6 | +8 | timing met at 100 MHz |

`--hexagon` builds use post-place and post-route `phys_opt_design`: without
it the final RTL missed by 71 ps in the unchanged tensor controller
(placement noise at 92 % slices).

Power (vectorless, routed): `npu_hx_0` 0.055 W dynamic in the overlay
(0.053 W out of context). The overlay's total dynamic power goes from
1.670 W to 1.947 W, but most of that difference is the unchanged tensor
accelerator's vectorless estimate shifting between builds (0.125 -> 0.344
W); see [subsystem.md](PPA/subsystem.md) for why the per-hierarchy number is
the fair comparison.

Multi-precision tensor unit (stand-alone, [tensor_mp.md](PPA/tensor_mp.md)):
the 4 x 4 INT8/INT16/FP16 array is 5.7x the LUTs of the INT8 array
(7,168 vs 1,266) at the same DSP count, 104.5 vs 126.6 MHz. A 16 x 16
version would need ~115k LUT (2.2x the Zynq-7020), so the deployed tensor
unit stays INT8 and FP16 runs on the board in the vector unit.

## Performance on the board

| Workload | Result |
| --- | --- |
| Scalar sort/sum program, 14,810 instructions | 57,314 cycles, CPI 3.87, L1I 99.97 % / L1D 99.85 % hits |
| INT8 saturating add, 512 elements | kernel 17.0 us (3.32 cycles/element), 23.5x numpy on the A9 |
| FP16 a*x+y, 512 elements | kernel 35.4 us (6.92 cycles/element), 5.4x numpy on the A9 |
| Tensor GEMM (unchanged unit) | all ISA checks match; up to 2.87 GMAC/s |
| SmolLM2-135M (W8A8 + AWQ) | correct answer; prefill 2.05 tok/s, decode 1.44 tok/s |

End to end from Python, one 512-element vector call took ~1.7-1.8 ms
because of host-side driver overhead (program load, flushes, MMIO
polling); the kernel is 17-35 us. Integration review round 3 reworked the
driver for resident programs and reused buffers (see below).

## Review rounds

Every component: one implementation agent, then three independent
review-and-improve agents. Findings that changed the design:

| Component | Round 1 | Round 2 | Round 3 |
| --- | --- | --- | --- |
| Memory (TCM, DMA, AXI arbiter) | AW-after-W and post-error test gaps (mutation) | datapath resets removed (FDCE 207 -> 33), port rename, TCM-port monitor | downstream ready no longer reaches the DMA through the arbiter |
| Scalar core | 5 illegal-encoding gaps; early fetch (CPI 4 -> 3) | stale early-fetch bug fixed; same-cycle responses; branch/vector CPI 3 | stop/start/reset gaps (6 mutants); no input->output comb path (Yosys proof) |
| Numerics + tensor PE | shared FP32 adder (-24 % LUT), FP16 mul stage rebalance | INT accumulate fused into DSP48, bit-exact with npu_pe | adder stage rebalance proven equivalent (Yosys miter) |
| Vector unit | busy over-run fix; single carry chain ALU (12 -> 8.5 ns est.) | false comb loop through tcm_gnt fixed; SRA precompute; VLEN=128 tested | clock enables (-15 to -31 % toggles); comb-path proof |
| Cache | read beats into the RAM (-42 % FF) | L2 tag compare registered; flush-without-invalidate gap | Icarus zero-time loop fixed; 4,160 stress seeds; multi-beat refill gap |
| Subsystem | 7 test gaps; load-data mux decoupled | profile; 1-cycle TCM/MMIO; safe STOP; unrolled kernels (-12/-14 %) | host-driver overhead; see the round-3 notes in the PR |

## Limitations and next steps

- The board bitstream predates subsystem review rounds 2-3 (STOP,
  1-cycle TCM/MMIO, driver rework); those are simulated and unit-tested,
  not yet measured on the board.
- FP16/INT16 tensor GEMMs are verified and measured stand-alone but not
  deployed (area). An INT8/INT16-only PE (no FP path, same DSPs) is the
  cheap step if INT16 GEMMs are needed.
- The scalar core and vector unit are not yet used by SmolLM2's software
  path: its GEMMs run on the tensor unit and the rest on the A9. Offloading
  elementwise layers (residual adds, requantisation) to the vector unit is
  the natural next step once the per-call driver overhead is low.
- Caches are not coherent with the A9 or the DMA; software flushes.
- Slices are at 92.5 %: further units need area back (e.g. the DSP-fused
  INT8 PE found in the numerics review would also shrink the 16 x 16 array).
