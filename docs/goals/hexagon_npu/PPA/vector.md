# PPA: SIMD vector unit

Commit `9534c3d`. Vivado 2026.1 out-of-context, `xc7z020clg400-1`, 10 ns,
vectorless power (exp-board run 38160036239). Before: no vector unit;
elementwise work ran on the Cortex-A9 (see [baseline.md](baseline.md)).

| Module | LUT | FF | DSP | BRAM | CARRY4 | WNS (ns) | Fmax (MHz) | Dynamic (W) | Total (W) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `npu_vector_unit` (VLEN 64, incl. 2 x fp16_add, 2 x fp16_mul) | 1,324 | 793 | 6 | 0 | 61 | 0.166 | 101.7 | 0.036 | 0.140 |

Performance (testbench, steady state):

| Case | Cycles per command | Elements per second at 100 MHz |
| --- | ---: | --- |
| Independent commands | 2.0 | INT8 400 M/s, INT16/FP16 200 M/s, INT32 100 M/s |
| Dependent chain | 5.0 | |
| VST after VST | 4.0 | |
| Paced by the scalar core (one command per 3 cycles) | 3.0 | INT8 267 M/s, FP16 133 MFLOP/s |

Kernels (testbench PERF lines, scalar-paced): 64-element INT8 residual add,
rolled loop 207 cycles, software-pipelined 100 cycles; 64-element FP16
a*x+y 429 / 247 cycles.

Share of the remaining overlay budget: 7.5 % of the 17,723 free LUTs, 6 of
the 26 free DSPs.

Timing: WNS +0.166 ns. The worst path is `x1_a` -> 8 LUT levels ->
`fmul1/s1_product` (the DSP input of `npu_fp16_mul`'s subnormal
normalisation). In the full overlay this unit's paths are independent of the
tensor array's, but the margin is thin; registering the multiplier operands
or moving the subnormal normalisation after the DSP is the planned fix.
