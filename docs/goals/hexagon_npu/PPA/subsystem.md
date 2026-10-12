# PPA: heterogeneous subsystem and the whole overlay

Commit `2c34b1d`. Vivado 2026.1, `xc7z020clg400-1`, 10 ns.

## Subsystem, out of context, with and without each unit

exp-board run 38169804215 (request `006-hexagon-board`), vectorless power.
Each row removes units with the `HX_ENABLE_*` parameters, so the
differences are what each unit adds in place (glue included).

| Configuration | LUT | FF | DSP | BRAM tiles | CARRY4 | WNS (ns) | Fmax (MHz) | Dynamic (W) | Total (W) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar + L1I + L1D + TCM + regs (no vector, L2, DMA) | 1,921 | 1,453 | 0 | 3 | 85 | 1.828 | 122.4 | 0.019 | 0.122 |
| No vector unit | 2,843 | 1,899 | 0 | 8 | 119 | 1.612 | 119.2 | 0.029 | 0.133 |
| No L2 (beat serializer instead) | 3,715 | 2,702 | 6 | 3 | 168 | 0.437 | 104.6 | 0.049 | 0.152 |
| No TCM DMA / AXI arbiter | 3,757 | 2,399 | 6 | 8 | 174 | 0.505 | 105.3 | 0.051 | 0.154 |
| All units | 4,186 | 2,774 | 6 | 8 | 188 | 0.389 | 104.0 | 0.053 | 0.157 |

| Unit added to the rest | LUT | FF | DSP | BRAM | Dynamic (mW) | Fmax effect |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Vector unit | +1,343 | +875 | +6 | 0 | +24 | 119.2 -> 104.0 MHz (its FP16 multiplier path) |
| L2 cache | +471 | +72 | 0 | +5 | +4 | none |
| TCM DMA + AXI4 arbiter | +429 | +375 | 0 | 0 | +2 | none |

## Whole overlay, routed in context

Before: the same build script without `--hexagon` (exp-board run
38188947073, request `011-baseline-overlay-power`; identical utilization and
WNS to the base branch's last build, run 37579935556).
After: the same build script with `--hexagon` (run 38169804215), which adds
`npu_hx_0` on GP0/HP0 next to the unchanged `npu_accelerator_0`.

| | LUT | FF | Slices | DSP | BRAM tiles | WNS (ns) | Failing endpoints |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | 35,477 (66.7 %) | 25,200 (23.7 %) | 11,141 (83.8 %) | 194 (88.2 %) | 26.5 (18.9 %) | +0.031 | 0 |
| After | 39,931 (75.1 %) | 28,279 (26.6 %) | 12,304 (92.5 %) | 200 (90.9 %) | 34.5 (24.6 %) | +0.063 | 0 |
| Delta | +4,454 | +3,079 | +1,163 | +6 | +8 | timing still met at 100 MHz | |

Power, vectorless `report_power` on the routed overlay:

| Hierarchy | Before (W) | After (W) |
| --- | ---: | ---: |
| Total on-chip | 1.813 | 2.099 |
| Dynamic | 1.670 | 1.947 |
| PS7 | 1.532 | 1.532 |
| `npu_accelerator_0` (RTL unchanged) | 0.125 | 0.344 |
| `npu_hx_0` | - | 0.055 |
| AXI DMA + interconnects | 0.013 | 0.016 |

Reading the power table: the subsystem itself costs 0.055 W dynamic in the
overlay, consistent with its 0.053 W out of context. The total rises by
0.277 W, but 0.219 W of that is the unchanged tensor accelerator, whose
vectorless estimate differs between the two builds: the baseline build
assigns its 194 DSPs and operand BRAMs almost no activity (DSPs and BRAM
< 1 mW in total), the `--hexagon` build 0.105 W and 0.028 W. Vectorless
power propagates default toggle rates through enables and the shared
interconnect, so it compares designs only roughly; the per-hierarchy
`npu_hx_0` number and the out-of-context table above are the like-for-like
comparison. A board-level power measurement needs a SAIF from simulation or
an external meter, which this goal did not include.

## Performance

| | Before | After |
| --- | --- | --- |
| Tensor GEMM | ISA path, 16 x 16 INT8 | unchanged; every board GEMM check matches ([board.md](../board.md)) |
| SmolLM2 decode | not measured on the base branch's overlay | 1.44 tok/s on the `--hexagon` overlay |
| Programmable control | none (Cortex-A9 only) | scalar core, CPI 3.87 on the board's sort program (57,314 cycles, 14,810 instructions) |
| Elementwise INT8 add, 512 elements | numpy on the A9: 400 us | vector kernel 17.0 us |
| FP16 a*x+y, 512 elements | numpy on the A9: 191 us | vector kernel 35.4 us |
| Peak vector throughput at 100 MHz | - | INT8 400 M elements/s, FP16 200 MFLOP/s (independent commands); scalar-paced 267 M / 133 M |

The subsystem's own Fmax (104 MHz out of context) is set by the vector unit's
FP16 multiplier; in the overlay the whole design still meets 100 MHz with
a slightly larger slack than before (the extra placement freedom of the
second cell did not hurt the tensor array's critical paths).
