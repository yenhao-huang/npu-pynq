# PPA: multi-precision tensor PE and array (INT8 / INT16 / FP16)

Commit `e9c950a`. Vivado 2026.1 out-of-context, `xc7z020clg400-1`, 10 ns,
vectorless power (exp-board run 38160036239). Before: the INT8-only
`npu_pe` / `npu_systolic_array` (baseline run 38148788279 for the PE; the
4 x 4 INT8 array measured in the same run as the multi-precision one).

| Module | LUT | FF | DSP | CARRY4 | WNS (ns) | Fmax (MHz) | Dynamic (W) | Total (W) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before: `npu_pe` (INT8) | 83 | 67 | 1 | 8 | 4.622 | 185.9 | 0.003 | 0.105 |
| After: `npu_pe_mp` (INT8/INT16/FP16, local FP32 adder) | 584 | 275 | 1 | 25 | 1.821 | 122.3 | 0.011 | 0.114 |
| Before: `npu_systolic_array` 4 x 4 | 1,266 | 1,000 | 16 | 128 | 2.099 | 126.6 | 0.026 | 0.129 |
| After: `npu_systolic_array_mp` 4 x 4 (FP32 adder shared by 3 PEs) | 7,168 | 3,112 | 16 | 200 | 0.432 | 104.5 | 0.072 | 0.175 |
| Delta, 4 x 4 array | +5,902 (5.7x) | +2,112 (3.1x) | 0 | +72 | -1.667 | -22.1 | +0.046 | +0.046 |

Performance per PE at 100 MHz:

| Precision | MAC/cycle per PE | 16 x 16 array peak |
| --- | ---: | ---: |
| INT8 (bit-exact with `npu_pe`) | 1 | 25.6 GMAC/s |
| INT16 (48-bit saturating accumulator) | 1 | 25.6 GMAC/s |
| FP16 x FP16, FP32 accumulation | 1/3 | 8.5 GFLOP-MAC/s |

Area per unit of throughput (4 x 4 array): INT8 79 LUT per MAC/cycle;
multi-precision 448 LUT per MAC/cycle in INT modes, 1,344 LUT per FP16
MAC/cycle.

Conclusion for the deployed overlay: a 16 x 16 multi-precision array would
need ~115k LUT (2.2x the device), so the deployed tensor unit stays INT8
and FP16/INT16 tensor work is a configuration for a larger part. FP16 runs
on the board in the vector unit ([vector.md](vector.md)). INT16 needs no
extra DSPs (the DSP48 multiplies 25 x 18), so an INT8/INT16-only PE (no FP16
path) is the cheap next step if INT16 GEMMs are needed.
