# Paper-based EDA tool exploration report

Current expanded status: [acceptance-status.md](acceptance-status.md). Fifty operations
are implemented, but whole-study acceptance remains incomplete. Earlier sections
record historical milestones; they are not a current completion claim.


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

## Binary linear transformations (operations 15, 29, 30)

Implemented inventory is 39. Parallel CRC consumes an explicit state and 64/128
MSB-first data bits; LFSR advances 32/64-bit state by 64/128 recurrence steps.
Alternatives are unrolled, independent matrix rows and shared XOR pairs. These
are architecture variants of two operations, not separately counted names.
Independent bit-step integer models cover all semantics, including zero state.

The planned compressor alias at ID 15 is replaced by exact affine equivalence:
it synthesizes both sources and propagates input coefficients plus a constant
through supported Yosys cells. Complete coefficient equality proves every binary
input, rather than assuming linearity from basis tests or trusting generator
matrices. Cycles, state, unknowns, nonlinear gates, multiple drivers and missing
wires are rejected. Positive/wrong-polynomial controls and an Icarus-replayed
counterexample passed, as did actual nonlinear rejection.

LFSR verify-only study 6455f0 passes affine and vector checks for both sizes;
its 64-bit SAT cross-check also passes, while the 32-bit SAT attempt times out.
CRC initial 1ca255 has both exact affine proofs; the 128-data-bit unrolled
simulation times out at 60 seconds and both SAT checks time out. The physical
study revision preserves all vectors and raises only simulation time to 300 s.
Proof kind and SAT outcomes remain distinct. No new PPA gain is claimed yet.

Completed saturation study c3969a: 32-bit throughput +21.58%, LUT growth 23.08%;
64-bit throughput +29.34%, LUT growth 28.00%. The latter exceeds the 25% resource
limit, so the family does not qualify. Its predeclared area objective stays fixed.
Barrel study 493114 has unchanged LUT counts, no 32-bit speed gain and only 2.21%
64-bit speed gain. It also does not qualify. All repeated records are retained.

## Iterative arithmetic checkpoint (42 operations)

Operations 22/31/44 add exact unsigned iterative multiplication, restoring
division and independent latency/throughput verification. Full-word interfaces
support parallel, one-bit-per-cycle and two-bit-per-cycle implementations.
The checker compares every ready/valid cycle, integer result, held output and
reset cancellation against an independent Python model. It verifies sustained
acceptance spacing before returning II. Division by zero is explicitly defined.
The setup cycle and two fixture observation cycles are separately accounted for.

Twenty-four substantial core/fixture checks pass at 16/32 bits, with 8192 random
stress cycles plus directed and sustained-throughput phases each. Mutations cover
wrong result, reset, backpressure hold, divisor-zero behavior, latency, II and
fixture observations. These are bounded sequential checks, not formal proof.
Parallel-to-serial area and serial-to-two-bit throughput comparisons are
predeclared at both widths, three physical pairs per case; runs are ongoing.

Argmax study 0402f6 is complete and is the fourth qualifying family. Sixteen
16-bit lanes improve throughput by 106.43%, with LUTs 480 -> 493 (+2.71%).
Sixteen 32-bit lanes improve throughput by 99.44%, with LUTs 944 -> 970 (+2.75%).
All three pairs pass the throughput resource gate at both sizes, with SAT and
vector correctness evidence. The throughput objective remains unchanged.

LFSR 2a7b3c completes both sizes and all pairs but does not qualify: LUT reduction
2.82% / 0%, throughput gain 13.11% / 6.40%. CRC 2a0316 also completes all pairs:
LUT reduction -1.22% / 1.27%, throughput gain 3.59% / -4.16%. Both studies retain
exact affine proofs, vectors and separate SAT outcomes; original failed attempts
remain preserved. These small or negative changes stay in the final denominator.

Eight operations, two additional qualifying families and the full all-case
acceptance audit remain unfinished. Four qualifying families alone do not prove
the aggregate objective gate or complete the goal.

Full focused validation: 290 passed, eight optional local-Yosys skips (the test
list in .github/workflows/ci.yml, --basetemp=.ic/pytest-expanded-15). All 24 real
arithmetic cycle experiments pass. make -C src/test lint sim passed; unchanged
hardware simulation targets are current. New-head hosted CI remains pending.

## Booth checkpoint (43 operations)

Operation 21 adds signed full-product multiplication with native, adjacent-bit
Booth and paired-bit Booth variants. Independent Python oracles pass for
8/16/17/32/64-bit operands, signed extrema, runs of ones and seeded random values.
Operands extend before negation; odd top groups repeat the multiplier sign bit.
Wrong sign-extension and negative-two recoding mutations fail verification.
Architecture variants count as one operation, not three.

The predeclared 16/32-bit native-versus-radix4 cases retain an area objective,
three paired physical runs and source-bound SAT/vector gates. Verification is
live as session 11253. Separate 8-bit formal controls are session 33010; these
controls do not meet the substantive benchmark requirement. No Booth PPA gain
is claimed. Multiplier/divider sessions 41858/63223 remain live.

Seventeen new oracle/mutation tests pass. The combined regression had 41 passes
and one Windows run-store directory rename PermissionError in an existing FIFO
test; that exact test passed on isolated retry. Published 1291564 hosted CI
passed. Seven operations (35,36,37,38,43,47,50), two more qualifying families
and complete whole-study acceptance remain unfinished.

Final focused regression rerun: 42 passed (--basetemp=.ic/pytest-booth-final).
The original Windows rename failure remains documented. Actual 8-bit SAT controls
are terminal: b345b5 proves equivalence; 894167 rejects incorrect sign extension
with an explicit proof failure (not timeout). Evidence: booth-formal-controls.json.
The 16-bit substantive proof b7f0ac timed out at 120 seconds; 32-bit is pending.
make -C src/test lint sim passed. Remaining account usage is 60%; no reset used.

Booth study 11253 is TERMINAL fc0266: both 16/32-bit vector checks pass,
both 120-second SAT proofs time out. Preserve booth-verification.json; no
physical gain qualifies. Small positive/negative controls remain separate.
Published signed multiplier checkpoint: 0c95fb5. Multiplier 41858 and divider
63223 remain active. Seven operations and full acceptance remain unfinished.

## Phase-controller checkpoint (44 operations)

Operation 36 adds a bounded cyclic phase FSM with preserved binary/one-hot
encodings. Synchronous reset selects phase zero; advance wraps modulo the
state count; disabling advance holds. Each phase has its own output bit.
The supported range is 4..256 power-of-two states. This is not a general
transition-graph compiler, and fault recovery from illegal state is not claimed.

The existing sequential_scoreboard operation now also accepts a phase contract.
Its independent integer model checks every post-reset cycle, all phase visits,
enable hold, wrap, reset and registered observations. No extra operation is
counted for the checker extension. Four negative controls break reset, hold,
direction and fixture output. Actual 64/128-state verification 0cf7ea passes all
eight core/fixture roles; evidence is phase-verification.json. No unbounded
sequential formal proof is claimed.

The physical study predeclares binary -> onehot, area objective, 64/128 states,
three matched Basic-flow pairs. Session 35612 is live. Report FF cost separately
from LUT decoding savings; do not assume a gain from encoding alone. Multiplier
41858 and divider 63223 also remain live; poll those exact handles before restart.

Relevant validation: 55 controller/FIFO/sweep/catalogue tests passed, then both
new phase pipeline gating tests passed (57 total). Repository lint/sim passed.
Published 0c95fb5 hosted CI passed. Six operations remain (35,37,38,43,47,50),
as do two additional qualifying families and the full all-case acceptance audit.

## Timing coverage checkpoint (45 operations)

Operation 43 audits actual routed clock coverage and report consistency. It
requires source/top identity, one clock, complete register coverage, no latches,
one setup path and complete check_timing sections. Mixed runs, missing reports,
contradictory counts, changed source, clock/slack and tool/build mismatches are
rejected. Report SHA-256 values are returned. Scope remains single-clock
register-to-register OOC setup; I/O delays, hold closure, board constraints and
protocol correctness are explicitly excluded. Negative setup slack is never
reported as meeting the requested period. Historical records without coverage
artifacts remain unaudited; no old record is silently upgraded.

Real experiments pass for 16/32-bit registered adders (50/98 sequential cells,
all clocked, all six timing checks clear). The intentionally unclocked-domain
design routes but fails audit: 65 total versus 64 clocked cells, one no_clock
pin and one unconstrained internal endpoint. Evidence: timing-audit-experiment.json.
Session 4544 is TERMINAL. The probe's installed Vivado help and actual report
format, rather than guessed syntax, were used to implement the parser.

Phase-controller study 35612 is TERMINAL ce1d2c and is the fifth qualifying
family under the declared LUT-area gate. All three pairs at 64 states reduce
LUTs 16 -> 0 and improve throughput 38.55%; at 128 states, 25 -> 0 and +46.02%.
DSP/BRAM remain zero. FFs increase 72 -> 130 and 137 -> 258, so this is a LUT
decode improvement with explicit register cost, not a reduction in all resource
classes. The throughput gate fails its FF-growth limit; the LUT-area gate passes
its separate stated conditions. The area objective remains unchanged.

Zero candidate LUTs make the exact benefit ratio unbounded. The repeated-run
summary now returns null for that exact scalar and a separately named finite
lower bound, using one LUT only in the ratio denominator. Actual resources,
100% LUT reduction and gate checks are unchanged. This conservative bound is
16x/25x for the controller cases, not a fabricated exact finite ratio.

Divider session 63223 is TERMINAL bf626e. Both 16-bit cases retain repeated
Vivado Synth 20-411 opt_design failures. Both 32-bit cases complete all pairs:
folding saves 95.13% LUTs but loses 38.39% throughput; two-bit iterations grow
LUTs 42.73% and lose 4.28% throughput. Neither qualifies. Original objectives
and every failed attempt remain preserved in divider-physical-study.json.

Multiplier 41858 remains live. Five operations remain (35,37,38,47,50), one
additional qualifying family and full all-case acceptance are still required.
Published d225871 hosted CI passed. Initial timing-focused suite: 54 passed,
four optional native-Yosys skips; zero-LUT/audit regression: 33 passed.
Full focused suite is running as 94470. Repository lint/sim passed. Remaining
account usage last checked: 58%; no reset credit used.

Final full focused validation: 341 passed, eight optional local-Yosys skips
(--basetemp=.ic/pytest-expanded-16). Real timing positive/negative controls pass.
Repository lint/sim passed. Fifth qualifying family is phase control; expanded
acceptance remains incomplete. Only multiplier session 41858 remains live.

Multiplier session 41858 is TERMINAL 18ccbd. All four cases and three pairs
complete. The predeclared serial-to-two-bit comparison qualifies at both widths:
16-bit throughput +44.29%, LUT reduction 2.80%; 32-bit throughput +61.88%, LUT
reduction 1.46%, with resource growth within the throughput gate. Source-bound
core/fixture cycle checks verify II=17 -> 9 and II=33 -> 17. This is qualifying
family six. The separate parallel-to-serial area comparisons remain tradeoffs:
LUT savings 67.18%/81.92% but throughput losses 86.56%/92.27%. Preserve all four
objectives/cases in the global denominator. No physical sessions remain live.
Five operations and the full all-case acceptance audit remain unfinished.

## Banked register-file checkpoint (46 operations)

Operation 35 adds conflict-aware two-read/one-write storage. The resettable FF
and banked distributed-RAM variants have identical externally visible bank
arbitration: low address bits choose a bank, read0 wins conflicts, rejected read1
returns zero, and read/write collisions return old data. RAM contents use
per-word validity bits to make synchronous reset logically clear every word.
There is no claim of true conflict-free arbitrary two-read access.

The independent logical-memory scoreboard checks every word, conflict priority,
read1-only requests, reset after writes, read-before-write and output latency.
Five mutation controls break conflict handling, reset masking, row selection,
collision ordering and registered output. The extension reuses operation 48;
it does not add another operation to the count. Both implementations support
one command batch per cycle; this is not two guaranteed completed reads.

Actual study ee1cbf passes all eight source-bound core/fixture roles at
16x64/two banks and 32x128/four banks. Each includes 8192 seeded stress cycles
plus full-address directed phases. Physical session 14940 is live with three
predeclared Basic-flow pairs per configuration and an area objective. No new
memory PPA win is claimed. New physical records include clock-coverage reports.

Validation: 77 relevant memory, phase, FIFO, sweep and registry tests passed.
make -C src/test lint sim passed. Published 3450cf9 hosted CI passed. Four
operations remain (37 skid buffer, 38 systolic tile, 47 ablation, 50 acceptance
audit). Six qualifying families are complete, but diverse full coverage and
all-case aggregate acceptance remain unproved. Remaining usage last checked:
56%, above the mandatory below-30% stop threshold.


## Backpressure pipeline checkpoint (47 operations)

Operation 37 generates one-slot elastic stages and two-slot skid stages that
cut combinational downstream readiness. Independent stage queues and an
end-to-end transaction queue check order, stalls, simultaneous transfer, reset
flush and drain. Sustained traffic calibrates minimum latency=stages and II=1.
The fixture adds two observation cycles, not an external handshake adapter.
Capacity is explicit: stages versus twice stages. Full skid stages have a
recovery bubble; no claim hides the extra storage or variable stalled latency.

Study 9a51bb passes all eight core/fixture checks at 32 bits/eight stages and
64 bits/16 stages, each with 8192 random cycles plus directed phases. Five
mutation controls detect reset, spill, capacity, architecture and fixture bugs.
The focused regression passes 96 tests. Physical session 12766 is live with
three predeclared Basic-flow pairs and a throughput objective. Register-file
session 14940 remains live. Neither pending study establishes a new PPA win.

Three operations remain: 38 systolic tile, 47 candidate ablation and 50 whole-
study acceptance audit. Six qualifying families are complete; full coverage,
ablations and the all-case aggregate remain unproved. Published 8e0c363 CI
passed. Account usage remaining is 54%; no reset credit was used.


## Ablation and completed memory checkpoint (48 operations)

Operation 47 computes complete binary factorial contrasts: per-setting effects,
averaged main effects and interactions through order four, in native resource,
throughput, latency and II units. It checks current source bytes, fixed nonfactor
configuration, matched physical conditions, equal repeats and unique executions.
Missing combinations are rejected; explicitly failed combinations are retained
without estimating effects. Three repeats describe reproducibility, not
statistical significance. Independent tests check known interaction functions,
zero LUTs, II normalization, repeat variation and twelve invalid designs.

Actual retrospective circular-FIFO storage ablations 987479/1bb522 separate RAM
policy from the shift-to-circular architecture change. Auto -> distributed adds
22/105 LUTs, removes 0.5 BRAM and changes throughput by +2.158/-38.287 million
transactions/s at 16x64/32x128. These are costs and conditional effects, not new
acceptance wins. Shift does not support distributed policy; no missing cell is
invented. Supplemental session 13279 measures width16/depth128 auto/distributed
with a predeclared area objective. Together with depth64 it will establish the
full storage-by-capacity interaction. Capacity is a workload factor, not an
optimization; the new case remains in the all-case denominator.

Register-file session 14940 is TERMINAL 7c99e5. Both configurations complete all
three pairs and qualify for LUT area: reductions 73.67%/83.31%, throughput changes
-4.02%/-2.08%, with no DSP/BRAM growth. All twelve source-bound timing audits pass.
This is qualifying family seven; conflict policy and command-batch throughput
remain explicit. The skid physical session 12766 remains live. Published
5f7d87f hosted CI passed. Initial ablation/physical/registry regression: 39 passed.
Full focused validation completed: 396 passed and eight optional native-Yosys
skips (--basetemp=.ic/pytest-expanded-18). Repository lint/sim passed.

Operations 38 (systolic tile) and 50 (whole-study acceptance audit) remain.
All-case aggregation, completed factorial experiment and final coverage audit
are still required. Expanded acceptance is incomplete. Usage last checked:
54% remaining, above the mandatory below-30% stop threshold; no reset used.


## Spatial matrix checkpoint (49 operations)

Operation 38 generates signed square matrix tiles with parallel arithmetic or a
true 2D systolic wavefront. Each PE forwards operands to its right/bottom neighbor
and accumulates one output; boundary injection is skewed. Exact output width is
2*element_width+ceil(log2(size)). The independent matrix oracle has no PE state
or injection schedule. The shared transaction checker verifies reset cancellation,
held results, every ready/valid cycle and calibrated batch latency/II. Nine
mutation controls detect forwarding, skew, signedness, accumulation, reset,
hold, latency, II and fixture errors. Dimensions 2..4 and widths 8..32 are bounded.

Study a971e4 passes all eight core/fixture checks for 2x2 matrices at 16/32 bits,
with 8192 stress cycles per role. Parallel minimum latency/II are 1/1; systolic
values are 5/5, and timing observations add two cycles. Throughput counts matrix
batches, not individual MACs. Physical session 34631 is live with a predeclared
area objective and three Basic-flow pairs. The focused matrix/arithmetic/sweep/
registry suite passes 83 tests. Published 6a00269 hosted CI passed.

Skid session 12766 is TERMINAL 0f05ab. Both configurations complete all three
pairs: throughput +3.79%/+20.52%, but LUT growth +2175%/+3423.33%. Neither qualifies
under the unconditional resource gates. All twelve timing-coverage audits pass.
FIFO-capacity session 13279 is TERMINAL b9190f. At fixed width16/depth128,
auto -> distributed increases LUTs 240.91% and reduces throughput 3.08%, while
removing 0.5 BRAM. All six timing audits pass; this is not a qualifying gain.

The completed source-bound storage-by-depth factorial analysis ac58d0 measures
an interaction of +31 LUTs and -11.038 million transactions/s: the storage-policy
cost grows with depth64 -> 128 at width16. Depth is a workload factor, not an
optimization. All old objectives and the supplemental area case remain in the
global denominator. Historical depth64 reports remain coverage-unaudited.

Seven qualifying families remain established. Only operation 50 (acceptance_audit)
is unimplemented; the whole-study objective aggregate and full coverage audit
remain unproved. The goal and PR remain incomplete. Last usage check: 53%
remaining, above the below-30% stop threshold; no reset credit used.

Final focused validation: 420 passed, eight optional native-Yosys skips
(--basetemp=.ic/pytest-expanded-19). Repository lint/sim passed. Matrix physical
session 34631 is the only remaining live experiment.


## Whole-study audit checkpoint (50 implemented operations; acceptance incomplete)

Operation 50 performs a bounded mechanical audit of the complete modular
configuration tree and committed study envelopes. It normalizes generator
parameters, preserves failed/missing cases, binds current core/wrapper bytes to
correctness and physical records, verifies cycle contracts and paired conditions,
and computes family coverage. The worst complete benefit across retries is used.
A missing ratio keeps the full aggregate undefined; failed candidates receive no
invented neutral value. The inventory checks 50 unique IDs/names and implementation
bodies, experiment READMEs, primary-paper references and test source files.

Actual audit f18ba5 finds 59 unique declared cases; 32 have complete verified
physical pairs. Structural inventory, all five architecture categories, at least
12 measured families and seven qualifying families pass. The full-case measurement
and aggregate-benefit gates FAIL. All-case geometric benefit is null, not an
average over only successful cases. Source files, declarations and evidence
inputs are hashed in acceptance-audit.json. Legacy reduction/FIFO declarations
remain in the denominator. Timing and cycle validation-control configurations
are explicitly identified as controls rather than candidate comparisons.

Structural inventory does not prove meaningful distinct purposes, PPA labels,
negative-test quality or actual exercise of every operation. These remain explicit
review items, as do historical report authenticity/coverage, declaration timing,
full historical inventory, documents and PR/usage state. The tool deliberately
returns completion_claim_supported=false; it is not a completion certificate.
The 50 implemented names do not fulfill the user's complete objective.

The focused acceptance/physical/registry suite passes 38 tests. They reject
missing declarations/evidence, changed sources, wrong proofs/cycle contracts,
duplicate measurements, synthetic PPA, changed objectives and favorable-retry
selection. They cover separate combinational core/wrapper identities and embedded
RTL strings in registry implementation inspection. Published f024667 CI passed.
Full focused validation completed: 435 passed, eight optional native-Yosys
skips (--basetemp=.ic/pytest-expanded-20). Repository lint/sim passed.

Live physical sessions: 34631 matrix tile, 20840 prefix adder, 39899 one-hot mux.
The latter two fill concrete gaps found by the audit, using existing declared
configurations and objectives. Proof timeouts and other physical failures still
need resolution; no acceptance gate is weakened. Last usage check: 51% remaining,
above the below-30% stop threshold; no reset credit used. PR #81 stays draft.
