# PPA: memory unit (TCM, TCM DMA, AXI4 arbiter)

Commit `70c5741`. Vivado 2026.1 out-of-context, `xc7z020clg400-1`, 10 ns,
vectorless power (exp-board run 38154769367, request `003-ppa-memory`).
Before: the design had no TCM, DMA or second-master arbiter (see
[baseline.md](baseline.md)); every number below is added area.

| Module | LUT | FF | DSP | BRAM tiles | CARRY4 | WNS (ns) | Fmax (MHz) | Dynamic (W) | Total (W) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `npu_tcm` 1024 x 64 | 73 | 130 | 0 | 2 | 0 | 3.995 | 166.5 | 0.013 | 0.116 |
| `npu_tcm_dma` | 204 | 191 | 0 | 0 | 14 | 4.058 | 168.3 | 0.003 | 0.106 |
| `npu_axi4_arbiter` | 194 | 138 | 0 | 0 | 0 | 5.256 | 210.8 | 0.002 | 0.105 |
| Sum | 471 | 459 | 0 | 2 | 14 | | | 0.018 | |

Performance (testbench, zero-latency AXI slave):

| Unit | Metric |
| --- | --- |
| TCM | 1 read or write per port per cycle, read latency 1; 2 x 8 B/cycle = 1.6 GB/s at 100 MHz |
| TCM DMA | 1024-word (8 KB) transfer in 1030 cycles each direction, 0.994 beat/cycle = 795 MB/s at 100 MHz; 4 bursts in flight |
| AXI4 arbiter | AR/AW +1 cycle, one address per 2 cycles per channel; R/W/B pass-through; 4010 R beats in 4049 cycles with both masters saturated |

Share of the remaining overlay budget (35,477 LUT used of 53,200, so 17,723
free; 26 DSP free): 2.7 % of the free LUTs, no DSPs.

Notes:

- `npu_tcm`'s 73 LUT / 130 FF beyond its two RAMB36: Vivado maps the RTL's
  port A onto the physical B port and puts a LUT3 on every read-data bit
  with a select register (the timing report's worst path is
  `mem_reg_0/DOBDO -> LUT3 -> a_rdata`). The cost is 0.14 % of the device and
  the TCM closes at 166 MHz, so it is recorded here rather than worked around.
- Review rounds: three (functional bugs: none; test gaps closed by mutation:
  AW-after-W slaves, post-error writes outside the descriptor, late and
  last-beat errors, request held while the FIFO is full, response codes,
  back-to-back start; timing: the downstream `arready`/`awready` no longer
  reaches the DMA through the arbiter).
