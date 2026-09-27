# Paper-based EDA tool exploration report

PR: [#81](https://github.com/yenhao-huang/npu-pynq/pull/81). Issue: [#80](https://github.com/yenhao-huang/npu-pynq/issues/80). Base: dev at
4a106811d4aaf104be99896606e68c6c8d20165c. Experiment date: 2026-09-28 (Asia/Taipei).

## Status: expanded goal in progress

The user expanded the target to 50 tools with substantial performance or area
gains across diverse microarchitectures. [acceptance-50.md](acceptance-50.md)
controls completion. The following is the historical first-milestone result and
does not satisfy the expanded acceptance.

## Initial milestone outcome

Ten new callable operations were implemented; eight directly support PPA
exploration. Each has a modular experiment. An original four-bit arithmetic
sharing case passed all 8192 binary input combinations. Local Vivado 2026.1
measurements show **8 -> 8 LUTs (0% area-resource improvement)** and
**8.133 -> 8.050 ns routed datapath delay (1.0205% reduction)**.

This is a small combinational experiment with automatic placement, not a measured
NPU improvement. The small delay difference may depend on implementation choices
and is not evidence of a robust architectural gain. Power was not measured.
The deliverable is a set of composable agent tools and a reproducible workflow,
not ten independently improved hardware designs or a full paper reproduction.

## Survey

[The survey](survey.md) covers primary RTLRewriter (ICCAD 2024), PPA-RTL and
Mascot/RTLOpt (DAC 2025), ASPEN (MLCAD 2025), and SymRTLO (primary preprint).
It links the papers and distinguishes their methods from the local adaptations.
No paper-reported performance percentage is reused as a local result.

## Tool results

| ID | Operation / category | Paper connection and purpose | Experiment result | PPA-related |
| --- | --- | --- | --- | --- |
| 01 | optimization_rules / lint | RTLRewriter, ASPEN, SymRTLO, Mascot: retrieve relevant rewrite guidance and semantic preconditions | Area/arithmetic query returns four attributed rules; power/pipeline rules remain queryable | Yes |
| 02 | width_advice / lint | RTLRewriter-inspired width-aware reasoning: bound add/subtract/multiply ranges | Sum of [0,15] and [0,15] is [0,30]; needs 5 unsigned bits, 11 fewer than a declared 16-bit result | Yes; advice, no measured gain |
| 03 | comb_check / debug | RTLRewriter fast-verification motivation: bounded exhaustive binary screening | All 8192 combinations agree; intentionally incorrect subtraction and X-output controls fail | Correctness support |
| 04 | ppa_measure / synth | ASPEN: actual EDA feedback instead of an RTL operator-count proxy | Two locally routed Vivado measurements; 8 LUTs each, 8.133/8.050 ns | Yes |
| 05 | ppa_provenance / synth | Support meaningful ASPEN-style measured comparisons | Compatible records accepted; mismatched version rejected in tests | Yes |
| 06 | ppa_compare / synth | Quantify actual resource/delay changes | LUT 0%, delay +1.0205%; FF percentage null because baseline is zero | Yes |
| 07 | ppa_pareto / synth | ASPEN: preserve multiple objective trade-offs | Candidate is nondominated in this two-point experiment; tests retain equal points and area/delay trade-offs | Yes |
| 08 | ppa_reward / synth | PPA-RTL: configurable PPA preferences | Equal LUT/delay weights give reward 0.005102668; 10 ns limit passes; failed correctness and violated limits are ineligible | Yes |
| 09 | ppa_select / synth | RTLRewriter: spend search budget on useful actions | Configured 90-second budget selects 60-second share-adder action; excludes 120-second action | Yes; configured history/costs |
| 10 | rtl_evaluate / pipeline/rtlrewriter | Compose verification, measurement and comparison | Positive rewrite checked and measured; negative control stops at correctness before any Vivado child | Workflow support |

Tools 01, 02 and 09 produce advice or decisions. Their outputs are not PPA
measurements. Tools 05-08 operate on the same real measurement pair. This is how
those eight operations compose into one optimization workflow.

The registry now exposes 18 operations (8 existing + 10 new). CLI, MCP, daemon,
and pi transport implementations were not edited. Operation input schemas select
the appropriate new backend. The existing lint/sim/synth defaults are preserved.

## Measurement contract

- Tool: Vivado 2026.1, SW Build 6511674; target xc7z020clg400-1.
- Flow: synth_design -> opt_design -> place_design -> route_design.
- Constraint: set_max_delay 10 ns, datapath_only, all inputs to all outputs.
- No fixed board pinout, board I/O constraints, activity data or clock-frequency
  measurement. The longest input/output path includes the inferred I/O path.
- Area field means LUT count, not physical silicon area; FF count is zero.
- No DSP/BRAM resource result or power result is claimed by this narrow flow.
- Metrics must share tool/version, part, constraint, stage and units to compare.
  External record provenance is declared metadata, not cryptographic authentication.
- Source hashes bind correctness checks to measurements; a changed source causes
  pipeline failure. Raw reports remain in local `.ic/`; compact evidence is committed.

| Metric | Baseline | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| LUTs | 8 | 8 | 0% |
| FFs | 0 | 0 | Undefined percentage |
| Routed datapath delay | 8.133 ns | 8.050 ns | 1.0205% |
| Power | Not measured | Not measured | Not available |

Source identities and runtime versions are in [tool 04 evidence](evidence/04.json).
The independent composed run is in [tool 10 evidence](evidence/10.json).

## Experiment loop and decision

| Hypothesis | Command/config | Observation | Decision / next step |
| --- | --- | --- | --- |
| Sharing arithmetic across a mux can help PPA | Tool 01; four-bit modular add/mux fixtures | Rule identified with width/signedness and timing preconditions | Check exhaustive behavior before measurement |
| Width analysis can expose unnecessary bits | Tool 02; [0,15]+[0,15], declared width 16 | Five bits suffice for the mathematical range | Advice only; prove workload bounds before narrowing production RTL |
| Rewrite preserves combinational behavior | Tool 03; 13 input bits | 8192/8192 binary vectors agree | Proceed to local routing |
| Shared adder reduces resources/delay | Tool 04; Vivado 2026.1, 10 ns | LUT count unchanged; 0.083 ns less delay | Reject an area-improvement claim; record the small timing observation |
| Measured feedback can guide selection | Tools 05-09 | Comparable records; candidate on frontier; positive gated reward | Retain the candidate for this case; use larger representative designs before generalizing |
| Composition enforces the gate | Tool 10; positive and deliberately wrong RTL | Wrong RTL stops before PPA; positive case measured independently | Pipeline boundary validated; sequential proof remains separate work |

Commands and configs are versioned under `exp/tool-exploration/`. The runner
supports individual tools, code/config/source/tool-version cache fingerprints,
atomic completion files and resumption. The account-usage threshold is enforced
by the agent workflow, not by the standalone runner. Usage remained above the
30% stop threshold (79% initially, 75% at the latest check).

## Validation

| Check | Result |
| --- | --- |
| Focused exploration, registry, CLI tests | 32 passed |
| Full tools/ic Python suite on Windows | 52 passed, 15 skipped, 1 failed, 1 setup error |
| Reproduction of those two failures on untouched dev | Same failures: Verilator executable discovery and existing Icarus wave handle |
| make -C src/test lint sim through MSYS2 | Passed; lint and 8 self-checking testbenches |
| Wheel build and contents | Passed; all seven new module files included |
| Skill validator | Passed; required local layout present |
| git diff --check / tracked-ignore check | Passed |
| OpenSpec CLI validation | Not run; CLI unavailable; proposal/design/tasks and requirement/scenario structure inspected |

The two Windows suite failures are pre-existing and were reproduced on dev at
4a106811. The Python interpreter cannot invoke the MSYS Verilator Perl entry
point as a native executable. The existing Icarus trace test returns no wave
handle on this host. These failures do not occur in the new combinational checker,
which owns its generated testbench. Skips reflect unavailable optional backends.
The new focused tests are added to the hosted CI job with Icarus; no Vivado step
was added to hosted CI. CI status must be read from the PR for its exact head.

## Limits and follow-up decisions

- No production NPU RTL, numeric model, AXI interface, register map or model format
  was changed. The experiment fixtures are outside the synthesized NPU tree.
- comb_check requires a complete, explicitly combinational interface, <=16 binary
  input bits and <=1024 output bits. It rejects known sequential/timed constructs
  but is not a full RTL parser, formal proof, latch analysis or X-input verification.
- No LLM model is trained or called by these deterministic tools. The reusable
  skill directs the agent's survey, extraction, implementation and experiment loop.
- No full e-graph optimizer, learned partitioner, symbolic FSM engine or C-MCTS
  framework is claimed. ppa_select uses a documented cost-normalized UCB adaptation.
- One tiny case does not establish benchmark-wide performance. The next useful
  study is a representative NPU datapath suite with sequential equivalence, fixed
  implementation constraints and activity-qualified power analysis.

## Reuse

Follow [reproduce.md](reproduce.md). The workflow skill is
[eda-tool-exploration](../../../.codex/skills/custom/ic_design/eda-tool-exploration/SKILL.md).
It includes the requested tool-count/topic and experiment-path questions, defaults
for no-question runs, acceptance criteria and the below-30% usage stop rule.

## Tool failure observed during validation

One final-validation pipeline attempt failed in local Vivado opt_design with
`ERROR: [Synth 20-411]` and no diagnostic detail (run 58a7fa; parent 33391c).
The pipeline returned a failed measurement stage and made no optimization claim.
The previously completed output had an older code fingerprint and was not reused
as final evidence. The failed tool-10 experiment was resumed independently and passed. Its positive
run reproduced 8 LUTs and 8.133/8.050 ns; its negative control failed at input
index 2 and launched only the checker child. All ten final evidence records share
the same code/config/source/environment fingerprint.

## Expansion checkpoint: physical and correctness foundations

Four additional registry operations are implemented: `clocked_ppa`,
`vector_equivalence`, `yosys_equivalence`, and `synth_adder_tree`. This is
14 implemented exploration operations, not 50 accepted tools. The reduction
generator's architecture variants count as one tool. Acceptance remains open.

Focused validation passed 42 tests before adding local-Yosys CI controls.
Actual container Yosys 0.23 proved a 16-bit four-input reduction, rejected a
wrong candidate, and rejected a mismatched input width. Larger SAT cases and
three-repeat physical studies are still running; no family is accepted yet.

Initial 16-bit, 16-operand results on Vivado 2026.1 build 6511674, Zynq-7020,
registered OOC boundaries, requested period 5 ns, latency=1 and II=1:

| Architecture | LUT | FF | DSP / BRAM | Estimated MHz | Assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| Serial baseline | 141 | 272 | 0 / 0 | 115.141 | Baseline |
| Balanced | 141 | 272 | 0 / 0 | 117.883 | About 2.38% throughput gain; below threshold |
| Carry-save | 372 | 272 | 0 / 0 | 166.472 | About 44.6% throughput gain, 163.8% more LUTs; tradeoff |

These are first-repeat observations, not accepted aggregate results. All three
miss the requested 200 MHz clock; the MHz column is inferred from setup slack,
not a timing-closure or board-performance claim. The carry-save candidate fails
the resource-growth gate despite its speedup. The next iteration must find a
better area/speed balance or a different architecture; neither candidate is
counted as a qualifying improvement.

## Network and sequential expansion checkpoint

Implemented exploration inventory: 23 operations. Nine additions after the
14-tool checkpoint are six independent network generators, architecture_sweep,
synth_fifo and sequential_scoreboard. Structural variants are not counted as
separate operations. The target remains 50 verified and exercised tools.

The network correctness study covers two substantial configurations per tool:

| Family | Configurations | SAT plus directed/random simulation |
| --- | --- | --- |
| Popcount | 64 / 128 bits | 64 proved; 128 SAT timed out |
| Priority encoder | 64 / 128 bits | Both proved |
| Leading-zero count | 64 / 128 bits | Both proved |
| Barrel shift | 32 / 64 bits | Both proved |
| Masked one-hot mux | 16 / 32 bits, 16 lanes | Both proved, including multi-hot total semantics |
| Stable argmax | 16 / 32 bits, 16 lanes | Both proved |

Independent Python-oracle tests cover both implementations, including zero,
all-one, walking-bit and 1,024 seeded random inputs. Oracle and RTL-to-RTL
checks are separate. Protocol scoreboards additionally passed shift and circular
FIFO implementations at 16x64 and 32x128, with 8,192 random cycles plus directed
fill/drain, replacement and reset phases. Ordering, capacity and reset bugs are
rejected by negative controls. FIFO physical results remain pending; its
latency is variable with a one-cycle minimum, not fixed.

The first priority-encoder 64-bit pair used 65 -> 50 LUTs, 71 -> 71 FFs, no
DSP/BRAM and 220.313 -> 239.406 estimated MHz. This is a 23.08% area decrease
and 8.67% throughput increase. The predeclared objective remains throughput;
no objective is changed after measurement. Full paired repeats and the 128-bit
configuration are still required before accepting the family.

The earlier reduction study completed all three 16-bit repeats. The 32-bit
first repeat was 285 LUT / 106.022 MHz serial, 285 / 109.337 balanced and
788 / 152.253 carry-save. The second 32-bit serial implementation failed in
opt_design with `[Synth 20-411]` and no explanatory message (run 4172b8).
The process stopped; its successful data and failed attempt remain preserved.
All four large reduction SAT attempts timed out. No reduction family PPA win
is accepted. A future retry must retain the failed attempt and matched source
and environment provenance.

## Clocked pair analysis and arithmetic expansion

Implemented inventory is 28 exploration operations. The five additions are
prefix-adder generation, CSD constant multiplication, shared multiple-constant
multiplication, multi-resource tradeoff analysis and paired-repeat summaries.
A source/metadata validator rejects inconsistent Fmax or throughput, changed
build/flow settings, reused report handles and changed repeated-run sources.
Repeated runs are reproducibility checks, not independent statistical samples.

Priority encoder has completed all three pairs at both 64 and 128 bits, with
SAT and vector checks bound to the measured sources. LUT reductions are exactly
23.0769% and 34.1615% across the repeats. Throughput increases are 8.6665% and
3.4468%; FF/DSP/BRAM use does not increase. Both configurations pass the LUT-area
gate. The predeclared throughput objective is retained for the eventual all-case
geometric mean. This is the first family with complete qualifying pair evidence;
whole-project acceptance still requires at least six and the remaining gates.
See evidence/priority-physical-study.json and priority-w64/128-paired.json.

Arithmetic correctness: 32/64-bit native versus Kogge-Stone addition proved
with SAT; 16/32-bit independent versus shared MCM also proved. CSD at width 16,
constant 255 proved, while width 32, constant 65535 remains inconclusive.
ABC normalization of the latter exceeded its time allowance; no proof is claimed.
Independent integer-oracle tests passed all arithmetic variants, including
maximum operands and coefficient packing. Simulation does not override missing
proof evidence.

Leading-zero Default optimization failed repeatedly for the 64-bit baseline
with Synth 20-411. Limiting Vivado to one worker did not cure it. A Basic flow
using explicit constant propagation and sweep successfully measured the first
64-bit pair, but a later attempt also failed. The alternative flow is explicitly
recorded and cannot be mixed with Default results. A binary-search narrowing
implementation is now an additional predeclared candidate; original tree cases
remain in the revised study. No leading-zero family win is accepted yet.

The initial 16-bit MCM pairs show 213 -> 106 LUTs and 170.387 -> 230.840 estimated
MHz. These are preliminary: 32-bit results and all repeats are still pending.

Timeout handling now uses an owned Windows Job Object or POSIX process group,
with a regression proving an unrelated sibling survives. Docker formal jobs
also use an internal timeout so terminating the Docker client cannot leave ABC
running. A controlled container test returned exit 124 and timed_out=true;
the timed-out job directory had no remaining process. This fixes experiment
resource accounting; it does not count as an additional exploration tool.

## Storage and netlist diagnostics checkpoint

Implemented inventory is 32 exploration operations. New diagnostics provide
source-bound Yosys profiles, endpoint fan-in/depth, fanout pin counts and memory
classification. All four ran on substantive FIFO and MCM structures. Their
estimates are not routed Vivado measurements.

MCM completed both 16/32-bit configurations and all three paired runs. LUT
reductions are 50.2347% and 52.2459%; throughput increases are 35.4801% and
33.7142%. All resource and correctness gates pass. This is the second family
with complete qualifying evidence, after priority encoder. Full parent results
and paired analyses are in evidence/mcm-physical-study.json and mcm-w*-paired.json.

The Yosys xc7 estimates show:

| FIFO configuration | Shift storage FF bits | Circular auto storage | Interpretation |
| --- | ---: | --- | --- |
| 16x64 | 1,031 | 25 FF bits + 6 distributed RAM primitives | Storage architecture changed |
| 32x128 | 4,104 | 55 FF bits + 1 RAMB18 primitive | BRAM substitution must be reported |

A distributed-only 32x128 candidate maps to 29 FF bits and 22 distributed RAM
primitives, with no BRAM in this estimate. Vivado must confirm the actual resource
tradeoff. MCM generic netlists reduce 16 arithmetic cells at seven structural
levels to six cells at two levels. These unweighted levels explain structure;
they are not measured timing delay.

FIFO timing now includes common input and output registers. An independent
queue trace verifies the two-cycle delayed fixture observations as well as the
core protocol. The fixture is a controlled measurement environment, not a drop-in
external FIFO protocol adapter. Empty/no-stall latency is reported as a minimum
(three cycles including the fixture), with II=1; latency can increase with queue
occupancy and backpressure. The measured source hash covers both core and fixture.

Initial 16x64 auto-inference physical results are 1,178 -> 19 LUTs and
1,068 -> 40 FFs, but BRAM grows 0 -> 0.5 tile. Therefore this is a resource
tradeoff, not an unconditional area win. A separate distributed-RAM revision
is predeclared and running. The auto study remains in the evidence and final
denominator; it is not replaced by the constrained candidate.

The pipeline now permits at most two implementation attempts by default and
records every attempt. A transient failure is retained even when a retry passes.
No repeated report can count as a fresh run. Leading-zero Basic revision b652b3
remains incomplete due to failed physical attempts and is retained separately.

## Mersenne residue reducer

Operation 32, `synth_constant_modulo`, brings the implemented inventory to 33.
It compares native unsigned remainder with balanced chunk sums and bounded
end-around folding for moduli 2**k-1. The output is canonical, including zero.
Parameters cover widths 8..128 and exponents 2..16 below input width. Native and
folded forms count as one operation; arbitrary/signed divisors are unsupported.

The independent Python oracle passes for 8/16/32/33/64/128-bit inputs, including
exhaustive small inputs, non-aligned chunks, modulus multiples and maximum
values. A deliberately incorrect zero canonicalization is detected. Relevant
arithmetic/catalogue/sweep tests: 57 passed. SAT and physical acceptance are
pending; this is not a third qualifying family. The declared cases are 16-bit
mod-15 and 32-bit mod-255, area objective, three pairs under the Basic flow.

First SAT study e2f022 proves 16-bit mod-15; 32-bit mod-255 times out internally
at 120 seconds (proof 587be6). The old timeout flag captured only process timeout;
the original artifact is retained and its log diagnosis is recorded. The checker
now classifies internal SAT timeout as well. A bounded 300-second retry is pending.
Focused tests including the timeout regression: 70 passed, four local-Yosys skips.

FIFO auto study d0d9f1 completes all pairs: 16x64 LUTs 1178 -> 19, throughput
+49.09%; 32x128 LUTs 4411 -> 22, throughput +72.88%. Both candidates add 0.5 BRAM
and therefore fail both unconditional improvement gates. This evidence illustrates
why dramatic LUT percentages cannot replace separate memory-resource accounting.

## Signed arithmetic kernels and completed FIFO evidence

Implemented inventory is 36 operations: FIR-window symmetry, signed dot-product
compression and saturating add/subtract sharing add three independent datapaths.
FIR is explicitly a window arithmetic kernel; sample-history and streaming
protocol logic are external. Dot product preserves full products and the complete
signed sum. Saturating ALU returns both clamped value and overflow for add/subtract.
The independent integer models cover extrema, negative coefficients, odd tap/lane
counts, subtraction and unsigned saturation. Mutated arithmetic/sign/overflow
outputs are rejected. Focused datapath/sweep/registry tests: 64 passed.

Distributed FIFO study 59e508 completes all three pairs for both configurations:
16x64 LUTs 1178 -> 41 (-96.52%), throughput +50.19%; 32x128 LUTs 4411 -> 127
(-97.12%), throughput +49.96%. DSP and BRAM remain zero, with fewer FFs. It passes
both area and throughput gates and is the third qualifying family, backed by
core and timing-fixture protocol checks. The separate auto-inference cases remain
resource tradeoffs and remain in the all-case denominator.

The 300-second 32-bit modulo proof retry 13fc7c is terminal with timed_out=true.
This corrects the timeout-classification behavior but does not establish proof.
Original 120-second evidence remains preserved. No modulo family win is counted.

FIR/dot-product SAT studies are in progress; initial 16-bit cases timed out.
Saturating ALU 32-bit SAT passed and its physical study is running. Barrel-shifter
and argmax physical studies now use matched Basic flow and two bounded attempts,
selected before their first physical measurements. Existing configurations and
throughput objectives are unchanged. None is counted as a new qualifying family
until both configurations, every repeat and the correctness gates are complete.

Completed FIR and dot verification studies each pass both large-interface vector
checks but time out in both SAT configurations (120 seconds per proof). Their
status is inconclusive, not verified equivalence; no physical improvement is
claimed. Full focused suite: 218 passed, six local-Yosys skips; repository gates pass.
