---
name: eda-tool-exploration
description: Survey primary EDA papers, extract composable AI-assisted RTL optimization tools, implement and test them through the IC registry, run modular PPA experiments, and prepare an evidence-backed report and PR. Use for paper-based EDA tool exploration or extending IC tools from ICCAD/DAC research.
---

# EDA tool exploration

Read the repository AGENTS.md and docs/rules before starting. Read this skill's
references/rules/{filetree,env,state-rules}.md. Reset STATE.md from the template
for a new run, or resume the existing state when explicitly requested.

## Scope

When parameters are missing and the user permits questions, ask together:
1. How many tools should be explored, and which topics deserve emphasis (PPA,
   formal verification, simulation, debugging)? Default: 10 tools, 8 PPA-related.
2. Where should the modular experiments live? Default: exp/tool-exploration/.

If the user says not to ask, apply the provided goal and these defaults directly.
Record count, topic priorities, paths, available EDA tools and the usage threshold.

## Workflow

1. Check remaining Codex usage before expensive work and between experiment
   batches. Remaining = 100 minus usedPercent. If any active window has less than
   30% remaining, save the exact resume point and stop. Do not consume a reset.
   If usage is unavailable, record that limitation; do not invent a percentage.
2. Survey primary papers (ICCAD, DAC and relevant related venues/preprints).
   Verify title, venue and method. Record citations and separate adaptations
   from full reproductions. Do not copy paper performance numbers as local results.
3. Claim one issue, create a dedicated worktree from dev, and prepare OpenSpec
   artifacts for major changes. Preserve other worktrees and user edits.
4. Define each tool's input/output contract, evidence scope, paper origin and
   acceptance test. Reuse the IC registry. Keep general tools in lint/sim/debug/
   synth and compositions in pipeline/<paper-abbreviation>/.
5. Implement scoped operations with bounded runtime and clear failed/unknown
   results. Add meaningful negative tests. Do not rewrite numeric or AXI contracts
   as a side effect of tool exploration.
6. For each experiment record: hypothesis -> config/command -> result -> decision
   -> next step. Use exp-tool-<id>-<paper>/ folders, common fixtures, exact source
   hashes, tool versions and conditions. Resume only matching completed inputs.
7. Verify candidates before expensive PPA runs. comb_check is only exhaustive
   binary simulation for explicitly combinational interfaces with <=16 input
   bits; sequential rewrites require a separate suitable equivalence workflow.
8. Compare compatible measurements; retain regressions, ties and Pareto trade-offs.
   Report LUTs as FPGA resources, routed datapath delay as delay, and missing
   power as unknown. Width reduction and synthetic test data are not measured PPA.
9. Write report.md (tool purpose, origin, results, limitations and decisions) and
   reproduce.md (environment, commands, expected evidence and failure handling).
10. Run focused tests plus repository gates. Inspect the complete diff, commit on
    the issue branch, and create a PR to dev if requested. Never merge by default.

## Acceptance and stopping

- Meet the requested tool count and PPA count; default 10 and 8.
- Every tool has a runnable experiment, primary-paper connection and recorded result.
- Include positive and negative correctness controls, provenance rejection,
  source binding, missing/zero metric handling and budget exhaustion.
- Deliver report, reproduction guide, modular experiments, skill and requested PR.
- On a real blocker, preserve work and report the exact unverified gate.
- A failed optimization experiment is useful evidence; do not manufacture a gain.
- Stop below the user's usage threshold even if acceptance is unfinished.

See references/workflow.md for the initial tool map and practical limitations.

## Expanded acceptance

When continuing the 0928 goal, use the complete acceptance-50.md contract under
repo docs/goals/0928-tool-exploration. Target 50 substantive tools and >=40 PPA
operations; require diverse, nontrivial microarchitectures and material measured
improvements. Never count the first ten-tool milestone as completion. Record
latency, II, FPGA resource trade-offs and all regressions. Keep the 30% stop rule.

For larger combinational studies, use `architecture_sweep` to bind generated
core and registered wrapper hashes to vector/SAT verdicts and routed records.
Keep the predeclared objective unchanged after seeing PPA. Retain failed cases
and their contribution to the final acceptance denominator. For FIFO studies,
use `sequential_scoreboard` with directed full/empty/reset phases and seeded
backpressure; a bounded queue-model pass does not establish formal equivalence.
Report variable latency explicitly and cover memory-read/handshake paths in the
physical fixture before claiming whole-FIFO throughput.

Validate completed physical pairs with `resource_tradeoff` and
`paired_repeat_summary`; compare source, build, clock, directive and worker/
optimization profiles. Distinct report handles are required for repeats.
PPA-only gates never replace source-bound correctness or the full family audit.
Record failed implementation attempts and retries separately. Optional AIG
formal normalization may spend time in ABC before SAT; preserve timeout as
inconclusive and ensure its processes terminate before further experiments.

Use `netlist_profile` followed by digest-bound `critical_cone`, `fanout_analysis`
and `memory_inference` to explain measured changes. Structural cell depth is not
physical delay. Memory primitives from Yosys do not replace Vivado utilization.
For FIFO sweeps, check both the core protocol and registered timing fixture;
the fixture is an observation environment rather than a deployable FIFO adapter.
Keep auto-inference and explicitly constrained memory policies as separate cases.
Bound physical retries and preserve every failed attempt in the study evidence.

Use `gf2_equivalence` only for its supported affine combinational subset. It
extracts exact expressions from actual netlists and must reject unknown, stateful
or nonlinear logic. Keep algebraic proofs and SAT attempts distinct, including
SAT timeouts. A generation-time matrix or passing basis-vector simulation is not
an affine proof. Raising a bounded simulation timeout must preserve vector count
and the original failed record; never silently reduce coverage for a PPA gate.

For iterative arithmetic, use `latency_throughput` on both the core and its
observation fixture. Require independent exact-integer result checks, reset
cancellation, held backpressured outputs and a sustained no-stall phase. Verify
each architecture's II rather than assuming identical cycle counts. Normalize
physical throughput by that II and include setup/observation latency explicitly.
Preserve parallel-to-serial area gains as tradeoffs when throughput falls beyond
the gate. Division-by-zero semantics must be identical across all variants.

For signed Booth multiplication, extend the multiplicand before negation and
test the most-negative operand and odd-width top recoding groups. Keep native,
radix2 and radix4 as variants of one operation. Small formal controls cannot
replace the predeclared substantial configurations or justify a PPA claim.

For phase-control encoding studies, compare the same number of states and
output phases, preserve each encoding explicitly, and check all states with an
independent integer oracle. Verify reset, enable hold, wrap and fixture latency.
Report register cost alongside LUT savings. A cyclic sequencer does not establish
arbitrary-FSM support, illegal-state fault recovery or lower power.

Audit new physical records with timing_constraint_audit and the exact measured
files/top. Keep the OOC setup scope explicit; coverage success is not requested
period closure, hold closure or board I/O validation. Reject mixed report runs
and missing coverage rather than upgrading historical evidence. For zero LUT
candidates, preserve the measured zero and FF cost. Use only the explicitly
reported conservative objective bound; the exact finite ratio is undefined.

For banked register files, keep conflict policy identical across implementations.
Check read-before-write, reset validity masking, every address and rejected-read
data/flags against a logical-memory oracle. Count command-batch II separately
from completed reads; conflict-limited memory is not a true unrestricted
multiport file. Preserve register/LUTRAM/BRAM costs as separate resources.


For skid/elastic pipelines, check both per-stage cycle behavior and end-to-end
transaction order. Calibrate no-stall latency and II, then stress full capacity,
backpressure, recovery and reset flush. Report differing capacities and all
storage costs. Never use the observation fixture as an external handshake
adapter or treat no-stall II as guaranteed throughput under arbitrary stalls.


For candidate_ablation, require the complete binary factor matrix and keep
nonfactor configuration fixed. Bind every cell to current source bytes and
independently verified semantics. Missing cells cannot be inferred; explicit
failures suppress effect estimation. Distinguish workload factors from changes
that preserve the same workload. Use native-unit conditional effects and
interactions to explain costs, not to revise objectives or manufacture extra
family wins. Reruns do not establish statistical confidence or causation.


For systolic matrix tiles, verify signed full-precision matrix semantics with
an oracle that does not model PE forwarding. Cover nonsymmetric matrices,
identity, extrema and every row/column product route. Count setup and wavefront
drain in latency/II, and use matrix batches as the throughput unit. Report all
DSP and storage costs. A throughput loss cannot be hidden by presenting MAC
frequency alone. Simulator coverage at larger sizes is not physical evidence.


Run acceptance_audit against the complete experiment/evidence roots. Never select
only successful study files or remove old declarations after observing results.
Inspect unmatched observations and unsupported declaration formats. Preserve
failed ratios as unknown and use the worst complete retry, not the best. Confirm
all remaining review items independently: structural registry/test-file presence
is not substantive tool or negative-test evidence. A null full-case aggregate
or completion_claim_supported=false cannot support a completed-goal claim.

Use `wide_workflow.py --tool all` to exercise tools 01-10 on the larger CSD16
study. Inspect explicit negative controls and bind core hashes to its formal
proof. Keep the workflow's combinational PPA records outside registered physical
acceptance. A cached or schema-valid output alone does not establish success.

When auditing an unclocked DSP C register, require complete static mux and
pattern-control evidence from the same routed run. Permit only the implemented
bounded configurations. An unused output path does not waive unconstrained
input endpoints or any other check_timing violation. Keep older missing-evidence
reports rejected and preserve failed development controls.

For ordered-bit SAT, prove the lowest bit without assumptions and allow each
later proof to assume only already-proved lower-bit equality. Require every
obligation in order, a clean exit, unchanged complete sources and a bounded
process. Verify lowest/highest-bit mutations and partial-source controls before
using a new proof mode to unlock physical experiments. Proof speed or success
does not count as a PPA gain.
