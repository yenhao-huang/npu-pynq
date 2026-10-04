# Count-based campaign results

The revised user instruction requires more than 20 experiments regardless of
cycles. All **21 distinct hardware configurations** have now passed functional
simulation (189 matrix cases total plus the abort/recovery cases). The lowest
measured full cold transaction is **579 cycles**, experiment 18. These are RTL
cycle measurements, not physical board latency or timing-closure claims.

All rows perform the same 16x256 by 256x16 multiplication: 8,192 input bytes,
65,536 MACs, 256 INT32 output words. Output width is a transport change and
uses TKEEP for row tails. Tail is the remaining compute/state-transition time;
compute also overlaps the B-load phase when enabled.

| Experiment | Input bits | Output bits | Boundary register | Overlap | Load | Tail | Output beats | Cycles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp01_ingress8 | 8 | 32 | 0 | 1 | 8192 | 34 | 256 | **8482** |
| exp02_ingress16 | 16 | 32 | 0 | 1 | 4096 | 34 | 256 | **4386** |
| exp03_ingress32 | 32 | 32 | 0 | 1 | 2048 | 34 | 256 | **2338** |
| exp04_ingress64 | 64 | 32 | 0 | 1 | 1024 | 34 | 256 | **1314** |
| exp05_ingress128 | 128 | 32 | 0 | 1 | 512 | 34 | 256 | **802** |
| exp06_no_overlap64 | 64 | 32 | 0 | 0 | 1024 | 289 | 256 | **1569** |
| exp07_no_overlap128 | 128 | 32 | 0 | 0 | 512 | 289 | 256 | **1057** |
| exp08_no_overlap32 | 32 | 32 | 0 | 0 | 2048 | 289 | 256 | **2593** |
| exp09_boundary64 | 64 | 32 | 1 | 1 | 1024 | 35 | 256 | **1315** |
| exp10_boundary128 | 128 | 32 | 1 | 1 | 512 | 35 | 256 | **803** |
| exp11_boundary32 | 32 | 32 | 1 | 1 | 2048 | 35 | 256 | **2339** |
| exp12_output64 | 64 | 64 | 1 | 1 | 1024 | 35 | 128 | **1187** |
| exp13_output128 | 64 | 128 | 1 | 1 | 1024 | 35 | 64 | **1123** |
| exp14_output256 | 64 | 256 | 1 | 1 | 1024 | 35 | 32 | **1091** |
| exp15_output512 | 64 | 512 | 1 | 1 | 1024 | 35 | 16 | **1075** |
| exp16_in128_out64 | 128 | 64 | 1 | 1 | 512 | 35 | 128 | **675** |
| exp17_in128_out128 | 128 | 128 | 1 | 1 | 512 | 35 | 64 | **611** |
| exp18_in128_out256 | 128 | 256 | 1 | 1 | 512 | 35 | 32 | **579** |
| exp19_distributed_a | 64 | 128 | 1 | 1 | 1024 | 35 | 64 | **1123** |
| exp20_dsp128 | 64 | 128 | 1 | 1 | 1024 | 35 | 64 | **1123** |
| exp21_dsp220 | 64 | 128 | 1 | 1 | 1024 | 35 | 64 | **1123** |

## Interpretation

For these tested configurations, the independently checked no-stall equation is:

```text
M*ceil(K/LANES) + K*ceil(N/LANES) + M*ceil(N/OUT_LANES)
  + M + N + 2 + BOUNDARY_REG + (OVERLAP ? 0 : K-1)
```

At the 579-cycle minimum, input still takes 512 cycles (88.43%), output 32,
and the remaining compute/transition tail 35. The experiment therefore changes
the magnitude of the bottleneck without eliminating input transfer dominance.

- Experiments 01-05 isolate ingress: 8,192 -> 512 loading cycles as input
  bandwidth grows from 8 to 128 bits per beat. All data is still transferred.
- Experiments 06-08 remove load/compute overlap at three widths. Each costs
  exactly 255 cycles, showing an independent scheduling benefit at K=256.
- Experiments 09-11 add one matched boundary pipeline stage at three widths.
  The cost is one cycle; its purpose is to split the BRAM/MUX/multiplier path.
- Experiments 12-15 increase output width with 64-bit ingress: output costs
  128/64/32/16 cycles while input remains 1,024 cycles. Marginal gains diminish.
- Experiments 16-18 combine 128-bit ingress and wider output, reaching
  675/611/579 cycles. The cost includes high BRAM occupancy.
- Experiments 19-21 retain experiment 13's transport and pipeline, changing
  A RAM implementation or DSP allocation. Each stays at 1,123 cycles; physical
  reports are needed to choose between these equal-cycle alternatives.

The lowest cycle result is 93.17% below 8,482 (14.65x cycle speedup). The
candidate is experimental: production wrapper/DMA/runtime integration and
on-board benchmarking have not been performed.

Selected physical validation is complete; all terminal outcomes are below.
The earlier experiment 04 has routed WNS -0.337 ns at 100 MHz. Experiment 09's
initial opt_design failure (Vivado Synth 20-411) is preserved, and its unchanged
checkpoint retry subsequently completed routing and passed internal timing.

## Synthesis comparison

All utilization below is synthesis of the standalone controller on
xc7z020clg400-1 with the same 10 ns clock constraint applied before implementation.

| Experiment | Synthesis LUT | FF | BRAM tiles | DSP |
| --- | ---: | ---: | ---: | ---: |
| exp04_ingress64 | 29854 | 17361 | 72 | 192 |
| exp09_boundary64 | 29806 | 17405 | 72 | 192 |
| exp13_output128 | 33434 | 17357 | 72 | 192 |
| exp18_in128_out256 | 39624 | 17635 | 136 | 192 |
| exp19_distributed_a | 34141 | 18372 | 8 | 192 |
| exp20_dsp128 | 37184 | 17355 | 72 | 128 |
| exp21_dsp220 | 32152 | 17354 | 72 | 220 |

Experiment 09 completed routing from its preserved synthesis checkpoint:
29,671 LUT, 17,407 FF, 72 BRAM tiles and 192 DSP. Final WNS is **+0.386 ns**,
hold slack **+0.132 ns**, pulse-width slack **+4.500 ns**, with zero failing
endpoints. Its full cold transaction is 1,315 cycles. The worst path now runs
from `load_inner_reg[8]` through write-control logic to A-bank RAM enable
(9.087 ns, 69.0% routing), rather than through the RAM/MUX/multiplier chain.

Experiment 19 keeps experiment 13's 1,123 cycles while replacing A BRAM banks
with distributed RAM: **64 fewer BRAM tiles**, at the cost of **707 LUT and
1,015 FF**. It uses only 8/140 BRAM tiles. This is a resource result; no routed
timing or board claim is made for experiment 19.

Experiment 18's 579 cycles requires 136/140 BRAM tiles (97.14%). This leaves
little BRAM capacity for integration despite fitting the standalone device.

## Final routed comparison

| Candidate | Cycles | Routed LUT | FF | BRAM tiles | DSP | Setup WNS at 100 MHz | Hold slack |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Experiment 04: 64-bit input | 1,314 | 29,781 | 17,376 | 72 | 192 | -0.337 ns | +0.136 ns |
| Experiment 09: add boundary register | **1,315** | 29,671 | 17,407 | 72 | 192 | **+0.386 ns** | +0.132 ns |
| Experiment 18: 128-bit input / 256-bit output | **579** | 31,501 | 17,635 | 136 | 192 | **-0.147 ns** | +0.145 ns |

Experiment 09 passes setup, hold and pulse-width checks. Experiment 18 has
10 setup-failing endpoints (TNS -0.674 ns), despite passing hold and pulse width.
Its worst path is active K -> A-bank RAM enable, 9.620 ns data delay with 68.2%
routing contribution. Thus the larger design shifts the critical path toward
memory control distribution. No additional design was introduced after 21.

The 579-cycle minimum is **not a timing-closed 100 MHz result**. The measured
100 MHz internal-timing alternative is experiment 09 at 1,315 cycles. These
remain standalone controller results; DMA/software integration and full-overlay
or board verification have not been performed.

## DSP allocation and final decision

With identical 64-bit input / 128-bit output, boundary stage and BRAM layout,
all three DSP budgets produce 1,123 cycles:

- 128 DSP: 37,184 LUT. Reserving 64 more DSPs costs 3,750 LUT versus 192 DSP.
- 192 DSP: 33,434 LUT. This is the fixed baseline for the resource ablations.
- 220 DSP: 32,152 LUT. Using the remaining 28 DSPs saves 1,282 LUT.

These are synthesis comparisons, not routed timing comparisons. The memory
ablation instead uses 1,024 LUT as distributed RAM and cuts BRAM from 72 to 8
tiles while keeping 192 DSP and 1,123 cycles; this is the strongest measured
BRAM reduction, but its timing is unchecked.

Stop at **21 completed, distinct experiments**, as the user requested.
Experiment 09 is the timing-closed 100 MHz controller candidate at 1,315 cycles;
experiment 18 is the 579-cycle research candidate with a setup violation;
experiment 19 is the BRAM-efficient 1,123-cycle candidate awaiting routing.
No 22nd configuration was designed or executed.

All 21 configurations passed RTL functional checks. This phase physically
profiled six selected configurations: 09/18 through routing, and 13/19/20/21
through synthesis. The existing routed experiment-04 evidence is also retained.
This scope does not establish timing closure for every tested configuration.

## Completion evidence

`python exp/0911_performance_optimization/audit_campaign.py` passed after all
physical processes terminated. It verifies 21 unique hardware parameter sets,
189 successful matrix cases, full target workload and phase counts, matching
frozen/canonical source hashes, command/temp-directory containment, the count
stop guard, per-experiment records, physical terminal reports, and the preserved
hardware archive. The machine-readable result is
`exp21_dsp220/artifacts/completion-audit.json`.

The earlier four-experiment stop and audit remain historical evidence under
`PHASE1.md` and experiment 04. They do not control the revised count-only goal.
