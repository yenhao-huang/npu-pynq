# PPA baseline: the design before #136

Every unit's PPA page compares against these numbers. Source: Vivado 2026.1
out-of-context synthesis + place + route on `xc7z020clg400-1`, 10 ns clock,
vectorless `report_power` (`src/hw/vivado_tcl/npu_matrix/ppa_module.tcl`),
exp-board run 38148788279, request `002-ppa-baseline`, commit `78a3da6`.

| Module | LUT | FF | DSP | BRAM tiles | CARRY4 | WNS (ns) | Fmax (MHz) | Dynamic (W) | Total (W) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `npu_pe` (INT8 PE) | 83 | 67 | 1 | 0 | 8 | 4.622 | 185.9 | 0.003 | 0.105 |
| `npu_systolic_array` 16 x 16 | 24,734 | 16,864 | 192 | 0 | 2,816 | 0.200 | 102.0 | 0.207 | 0.313 |
| `npu_accelerator` 16 x 16 | 32,333 | 21,265 | 194 | 24 | 3,338 | -0.026 | 99.7 | 0.339 | 0.449 |

The full overlay (block design with PS7, interconnects and AXI DMA) from the
base branch's last build (exp-board run 37579935556, in-context, routed):
35,477 LUT (66.7 %), 25,200 FF (23.7 %), 194 DSP (88.2 %), 26.5 BRAM tiles
(18.9 %), 11,141 slices (83.8 %), WNS +0.031 ns at 100 MHz.

Notes:

- Out of context, the accelerator misses 100 MHz by 26 ps; in the block design
  it meets timing with 31 ps to spare. Both say the same thing: the tensor
  path has no margin, so the new units must not share its critical paths.
- Static power (~0.10 W) is the device's and is the same for every module; the
  dynamic column is the one to compare.
- The "Performance" of the INT8 tensor unit: 256 MAC/cycle x 100 MHz =
  25.6 GMAC/s peak (51.2 GOPS).
