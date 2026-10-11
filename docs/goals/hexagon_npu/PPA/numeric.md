# PPA: FP16/FP32 arithmetic

Commit `e9c950a`. Vivado 2026.1 out-of-context, `xc7z020clg400-1`, 10 ns,
vectorless power (exp-board run 38160036239, request
`005-ppa-numeric-vector`). Before: the design had no floating-point
arithmetic (see [baseline.md](baseline.md)); these are added area, and they
are instantiated inside the vector unit and the multi-precision PE.

| Module | LUT | FF | DSP | BRAM | CARRY4 | WNS (ns) | Fmax (MHz) | Dynamic (W) | Total (W) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `npu_fp16_mul` (latency 2) | 180 | 40 | 1 | 0 | 6 | 0.799 | 108.7 | 0.004 | 0.107 |
| `npu_fp16_add` (latency 3) | 186 | 83 | 0 | 0 | 10 | 2.939 | 141.6 | 0.004 | 0.107 |
| `npu_fp32_add`, `SUBNORMALS=0` (latency 3) | 310 | 147 | 0 | 0 | 25 | 1.284 | 114.7 | 0.006 | 0.108 |

Performance: each unit is fully pipelined, one operation per cycle:
100 MFLOP/s per unit at 100 MHz. Accuracy: IEEE 754 round-to-nearest-even,
subnormals supported (the FP32 accumulator configuration has none by
construction), canonical NaN; bit-exact with the numpy golden model on
>20k random + directed vectors per unit.

Timing note: `npu_fp16_mul` has the least slack (0.8 ns). Its stage 1
normalises subnormal significands before the DSP multiply; that path also
limits the vector unit (below). It is the first candidate for the
post-integration improvement pass.
