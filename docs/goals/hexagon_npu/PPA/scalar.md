# PPA: scalar control core

Commit `87c5d6d`. Vivado 2026.1 out-of-context, `xc7z020clg400-1`, 10 ns,
vectorless power (exp-board run 38159534136, request `004-ppa-scalar`).
Before: the design had no programmable control (the Cortex-A9 sequenced
every job), so the whole core is added area; see [baseline.md](baseline.md).

| Module | LUT | of which LUTRAM | FF | DSP | BRAM | CARRY4 | WNS (ns) | Fmax (MHz) | Dynamic (W) | Total (W) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `npu_scalar_core` | 950 | 44 | 392 | 0 | 0 | 49 | 3.086 | 144.6 | 0.009 | 0.112 |

Performance (ISS cycle model, checked cycle-exactly by the testbench):

| Instruction class | Cycles at a 1-cycle memory | General (memory latency L) |
| --- | ---: | --- |
| ALU, LUI, AUIPC, JAL, JALR, branch, vector command | 3 | max(3, L + 2) |
| Load, store | 5 | 4 + L |
| VFENCE | 5 + cycles `vec_busy` is high | |

At 100 MHz with an L1 hit (L = 1): about 30 MIPS on ALU/branch code,
20 M memory operations per second. Each vector command takes 3 core cycles,
so the core can feed the vector unit one command every 30 ns.

The routed worst path is `state` -> 2 LUTs -> `i_req_addr` (the
fetch-address select, an output of the core); the core's internal paths have
more slack. Yosys put the core at 992 LUT / 402 FF; Vivado's 950 / 392 agree.

Share of the remaining overlay budget (17,723 free LUTs, 26 free DSPs):
5.4 % of the free LUTs, no DSPs.

Review rounds: three. Functional bug found and fixed: a stale early fetch
when a load/store then faulted (round 2). Test gaps closed by mutation:
illegal encodings (5 survivors in round 1), stop/start/reset interactions
(6 survivors in round 3). CPI went from 4/6/5 (ALU, load/store, vector) to
3/5/3. Same-cycle responses were added and a Yosys check confirms that no
input reaches an output combinationally.
