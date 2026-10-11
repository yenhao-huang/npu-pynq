# PPA: L1 I$, L1 D$, L2 cache, arbiter and AXI line bridge

Commit `a23a035`. Vivado 2026.1 out-of-context, `xc7z020clg400-1`, 10 ns,
vectorless power (exp-board run 38169804215, request `006-hexagon-board`).
Before: no cache (see [baseline.md](baseline.md)); all area is added.

| Module | Configuration | LUT | FF | DSP | BRAM tiles | CARRY4 | WNS (ns) | Fmax (MHz) | Dynamic (W) | Total (W) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `npu_cache` L1 I$ | 1 KB, direct mapped, read-only | 151 | 130 | 0 | 0.5 | 18 | 4.261 | 174.2 | 0.002 | 0.105 |
| `npu_cache` L1 D$ | 1 KB, direct mapped, write-back, CPU_HOLD | 195 | 387 | 0 | 0.5 | 18 | 4.068 | 168.6 | 0.004 | 0.107 |
| `npu_cache` L2 | 8 KB, 2-way LRU, 256-bit CPU side, 64-bit beats | 542 | 115 | 0 | 5 | 20 | 3.264 | 148.5 | 0.009 | 0.112 |
| `npu_mem_arbiter` | 2 masters, round-robin | 156 | 8 | 0 | 0 | 0 | 5.970 | 248.1 | 0.001 | 0.103 |
| `npu_axi4_line_bridge` | 32-byte line = 4-beat INCR burst | 81 | 137 | 0 | 0 | 0 | 6.250 | 266.7 | 0.001 | 0.104 |
| Sum | | 1,125 | 777 | 0 | 6 | 56 | | | 0.017 | |

The standalone arbiter is generic (two read/write ports); in the subsystem
port 0 is the read-only I$ (`P0_READ_ONLY`), which removes most of its
256-bit write mux.

Performance (testbench, zero-wait memory):

| Event | Cycles |
| --- | ---: |
| L1 hit (I or D) | 1 |
| L1 miss, L2 hit | 15 |
| L1 miss, L2 miss | 24 |
| L1 dirty miss, L2 hit | 28 |
| Accept rate per cache | one request per 2 cycles (never the bottleneck: 0 stall cycles in the subsystem profile) |

Effect on the scalar core (subsystem profile, case b, deliberate L1 D$ set
conflicts): CPI 5.86 = 3.30 core + 0.30 L1 D$ hit latency + 2.14 L1 D$
misses + 0.12 L1 I$ misses; a working set that fits in L1 D$ runs at about
CPI 3.6. Without the L2 (`HX_ENABLE_L2 = 0`) the same program takes 6361
instead of 5931 cycles (+7 %) with an ideal memory; on the board's DDR the
gap is larger (measured in [subsystem.md](subsystem.md)).

Review rounds: three. No hardware bug found; refill registers removed
(-42 % FF), L2 tag compare registered (cross-module 8-9 ns path removed),
W data registered in the bridge, an Icarus zero-time loop fixed (outputs
assigned once per always_comb, proved equivalent with Yosys), 4,160 random
stress seeds with flushes and resets, two real test gaps closed
(flush-without-invalidate, multi-beat refill word).
