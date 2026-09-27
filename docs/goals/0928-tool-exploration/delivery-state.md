# GitHub delivery state

Run ID: 2026-0928-issue80-a
Started: 2026-09-28 Asia/Taipei
Repository: yenhao-huang/npu-pynq (configured remote npu_in_pynq redirects here)
Issue: https://github.com/yenhao-huang/npu-pynq/issues/80
Base branch: dev, 4a106811d4aaf104be99896606e68c6c8d20165c
Head branch: npu/issue80-a
Scope: implement and validate goal.md; publish one PR to dev; no merge requested.

| Step | Status | Evidence |
| --- | --- | --- |
| Confirm repository and authorization | completed | Goal requires PR; remote and gh identity checked |
| Search and claim issue | completed | No matching prior issue; #80 assigned to repository owner and agent a claim verified |
| Read contribution rules | completed | AGENTS.md and docs/rules reviewed |
| Inspect branch and diff | completed | Entire change reviewed against dev; no production RTL or unrelated user changes included |
| Validate and commit | completed | d5951b4; 32 focused tests; lint/sim; ten final matching experiments; baseline Windows limitations recorded |
| Draft PR | completed | .ic/pr-body.md contains summary, exact tests, measured result and limitations |
| Push, create and verify | completed | https://github.com/yenhao-huang/npu-pynq/pull/81; open, non-draft, base dev, head npu/issue80-a |
| Handoff | completed | PR attached to the chat; report/reproduce/skill and ten evidence records committed |

Remaining usage: 75%; stop below 30%.
Known validation limitations: two full-suite Windows failures reproduced on dev;
OpenSpec CLI is not installed. Raw evidence is in .ic and compact summaries are
under docs/goals/0928-tool-exploration/evidence/.

CI was in progress when the PR was created. Consult the PR for the exact final head status.

## Expanded user goal (supersedes initial completion)

The user now requires fifty substantive tools and diverse microarchitectures with
clear measured performance or area improvement. Follow acceptance-50.md. The old
10-tool milestone and 1.02% result do not satisfy this goal. Resume development on
the same issue branch and PR. Status: in_progress. Remaining usage at expansion:
74%. Prior turn classification: progress (ten tools, actual measurements and PR).

### Running expansion experiments

Four foundation operations are now implemented (14 total exploration ops).
Focused tests, including 16/32-bit reduction simulations and interface negative
controls, are in tools/ic/tests/test_wide_exploration.py. CI now includes them
and installs open-source Yosys; Vivado remains local only.

Active processes at this checkpoint:
- Physical study: exec session 14429, exp-tool-14-rover/study.py; three repeats
  for serial/balanced/compressor at (width,lanes)=(16,16),(32,16). Do not start
  another copy until this process completes. Its namespace reflects source
  content at launch; subsequent validation-only edits do not change that run.
- Formal study: exec session 75065, exp-tool-13-rover/check.py with existing
  codex-sandbox-agent-workspace container. Each SAT attempt has a 120 s limit.

The current physical runner's resume key does not include environment versions;
that must be corrected before final reproducibility acceptance. Large formal
proof results, complete paired repeats, 36 additional substantive tools,
sequential/memory families and full acceptance remain pending. First-repeat
reduction data show no unconditional qualifying win (see report.md).

Latest focused validation: 43 passed, two local-Yosys tests skipped; actual
container controls passed separately. Repository make lint/sim passed, with
unchanged simulation outputs already up to date. The 16-bit/16-operand balanced
SAT attempt timed out after 120 s (run 9ca867), confirmed through the
proof-log artifact handle. Do not replace inconclusive proof with a claim
of formal acceptance.

### Network and sequential checkpoint

Implemented inventory is now 23 operations. Tests cover six distinct network
families, a source-bound/environment-aware architecture sweep, FIFO generation
and cycle-exact queue scoreboarding. Physical acceptance remains incomplete.

Authoritative process state at this checkpoint:
- Old physical reduction session 14429 is TERMINAL (exit 1), with failure
  4172b8 on w32-n16-serial-r1. opt_design returned Synth 20-411 without detail.
  Preserve the old output namespace and all successful repeats; do not blindly
  restart study.py into a new digest and discard the denominator.
- Formal reduction session 75065 is TERMINAL. Four-operand 16-bit controls
  proved; both 16/32-bit 16-operand architectures timed out at 120 seconds.
- Network verify-only session 53625 is TERMINAL: 11/12 configurations proved,
  with only 128-bit popcount not proved. All vector checks passed.
- Priority physical session 3537 is LIVE. It runs network_study.py --tool 24
  with existing codex-sandbox-agent-workspace. Poll before launching more
  priority studies. Partial per-pair JSON lives under .ic/studies/synth_priority_encoder.

New experiment folders: 23-rover, 24/25-prefixrl, 26/27/28-rtlrewriter,
34-scalar, 48/49-rtlrewriter. FIFO correctness experiment completed all four
variant/config combinations; physical timing still needs a common registered
fixture. Do not call FIFO latency fixed or use internal-only paths to claim
whole-FIFO throughput. Current test suite: 91 passed, 2 skipped before the small
simulation-error rejection refinement. Previous published commit CI is green.

Remaining scope: 27 substantive operations, at least 12 diverse families total,
six qualifying family wins, complete repeat/proof coverage, all-case geometric
mean and final acceptance audit. The inventory's planned compressor-only tool
must be replaced because carry-save is already a variant of synth_adder_tree;
counting an alias would violate the acceptance contract.

Physical progress update: priority 64-bit has all three matched repeats with
65 -> 50 LUTs and 220.313 -> 239.406 estimated MHz. The first 128-bit pair is
161 -> 106 LUTs and 210.881 -> 218.150 estimated MHz. Remaining 128-bit repeats
are live in session 3537. Leading-zero study is independently live in session
47144 (network_study.py --tool 25); poll these handles before launching duplicates.
No family is marked accepted until its complete evidence has been audited.

## Arithmetic and physical-analysis checkpoint

Implemented count: 28 (22 operations remain). New IDs: 16 prefix adder,
17 CSD multiplier, 18 MCM, 45 resource tradeoff, 46 paired-repeat summary.
No lifecycle helper is counted as a tool. The compressor alias in planned ID15
still requires replacement by a distinct useful operation.

Priority session 3537 is TERMINAL success, parent 4b7973; both configurations
pass all three area pairs. Full evidence is preserved in
priority-physical-study.json and priority-w64/128-paired.json. This establishes
one qualifying family, not global acceptance. The objective remains throughput.

Leading-zero original session 47144 is TERMINAL with measurement failures.
Single-worker retry session 19794 is TERMINAL, parent 3957a1: w64 fails in
opt_design; w128 completes but does not achieve 15% gain. Errorinfo confirms
Synth 20-411 without further detail. All these attempts remain in the run store.

Live processes at this checkpoint (poll authoritative handles first):
- 23666: leading-zero Basic revision, four predeclared cases (original tree and
  binary-search candidate at 64/128 bits). First Basic w64 baseline passed;
  candidate r1 later failed, so inspect retained failures and remaining cases.
- 31080: MCM 16/32-bit three-pair physical study. First two 16-bit pairs improve
  LUTs 213 -> 106 and estimated MHz 170.387 -> 230.840. Remaining results pending.

Arithmetic verification session 1105 is TERMINAL. Prefix cases prove (19db79);
CSD 16 proves / 32 is inconclusive (8db0bc); both MCM configurations prove
(fa87aa). AIG normalization attempt 6f7ec7 timed out in ABC. Its orphan processes
were explicitly identified by run directory and terminated. The formal backend
now uses container-internal timeout, and process.run uses Windows jobs / POSIX
groups to terminate owned children. Controlled guarded attempt 994c19 returned
124 with timed_out=true; neither timed-out job directory retained a process.

Windows parent-only/taskkill cleanup failed its first adversarial test due to
restricted process enumeration. The replacement Windows Job Object test passes,
including a TERM-ignoring child and an unaffected sibling. This change touches
the shared process helper; rerun the full focused suite and CI before acceptance.

Latest verification: 135 passed, four optional local-Yosys tests skipped.
Container AIG controls pass separately (positive ac02b3, negative 0fea7e).
The timeout helper regression passes on this Windows host. These results
include the final shared-process cleanup implementation, not only the earlier
parent-only cleanup attempt.

## Registered storage and diagnostic checkpoint

Implemented inventory: 32 operations (18 remain). IDs 39-42 now have real
Yosys experiments and unit tests. MCM session 31080 is TERMINAL success, parent
6e132a: both configurations pass area and throughput gates for all three pairs.
It is the second complete qualifying family. Priority remains the first.

Netlist study session 97218 is TERMINAL success. Both FIFO sizes and mappings,
and all MCM graphs, were analyzed. Pilot 22343 also completed. The xc7 estimator
finds BRAM substitution in the 32x128 auto circular FIFO; a distributed policy
was added and verified to avoid BRAM in the structural estimate (profile 3366da).

Leading-zero Basic session 23666 is TERMINAL, parent b652b3. Original w128 tree
completed but misses the threshold; w64 and both binary-search cases have
measurement failures. Preserve leading-zero-basic-attempts.json. New sweeps
now record bounded retries (default two attempts); the old study had one attempt.
A future repair must retain all prior failures and existing source/flow evidence.

Live physical studies at this checkpoint:
- 69500: FIFO auto inference, two configurations, three pairs, Basic flow,
  two attempts maximum. Initial 16x64 candidate uses 0.5 BRAM, so cannot pass
  the unconditional area gate despite a large LUT reduction.
- 21483: FIFO distributed-RAM revision, two added configurations, same measured
  timing fixture, Basic flow and retry policy. Original auto data remains retained.

The FIFO fixture has matched registered boundaries and a separately verified
queue trace delayed two cycles. It is a timing environment, not a deployed
external ready/valid adapter. Minimum latency is three cycles including fixture;
II=1. Fixed and minimum latency kinds cannot be mixed in a paired comparison.

Remaining full-goal work includes 18 distinct operations, at least one additional
microarchitecture family beyond the current eleven generator families, four more
qualifying family wins, missing proofs/measurements, ablations and the final
all-case acceptance audit. Do not claim completion from 32 tool names.

Validation for this checkpoint: 154 focused tests passed, six tests requiring a
local Yosys executable skipped. Actual container netlist experiments passed
separately. `make -C src/test lint sim` passed. Remaining usage is 66%.

### Mersenne operation implementation

Operation 32 now emits native/folded canonical Mersenne reducers; inventory is 33
with 17 operations remaining. Independent integer-oracle and mutation checks
passed (57 arithmetic/catalogue/sweep tests). Verify-only session 5034 is active;
poll its status before restarting. Cases 16-bit mod-15 and 32-bit mod-255 are
predeclared under exp-tool-32-mersenne/config.json. No physical win is claimed.

The first modulo study is terminal (e2f022): 16-bit mod-15 SAT passed; 32-bit
mod-255 hit Yosys's internal 120-second SAT limit (587be6/proof.log). The original
record says timed_out=false because only outer-process timeouts were classified;
the actual log explicitly reports proof timeout. The implementation now recognizes
this internal timeout, with separate timeout/failure regression controls. Original
evidence remains unchanged. A bounded 300-second proof retry is session 80622.
Relevant tests including this correction: 70 passed, four local-Yosys skips.

FIFO auto study 69500 is terminal success (d0d9f1), all three pairs at both sizes.
LUT reductions are 98.39%/99.50% and throughput gains 49.09%/72.88%, but both
candidates add 0.5 BRAM tile, so both unconditional PPA gates fail. Paired results
and all attempts are committed; distributed session 21483 is still running.

Latest full focused validation: 173 passed, six optional local-Yosys skips
(`--basetemp=.ic/pytest-expanded-12`). Repository lint/sim passed; unchanged
hardware simulation targets are current. The 33-operation checkpoint remains
incomplete against the 50-operation contract.

## 36-operation checkpoint

New operations: synth_fir, synth_dot_product, synth_saturating_alu (IDs 19/20/33).
Fourteen distinct operations remain. FIR/dot products are full-precision signed
kernels with independent integer oracles; PPA and proof failures are not gains.

Distributed FIFO 21483 is TERMINAL success, parent 59e508. Both configurations
pass area and throughput gates in all three pairs, without DSP/BRAM growth.
This is the third qualifying family (priority, MCM, FIFO). Auto inference remains
a failed unconditional gate and is preserved. Modulo retry 80622 is TERMINAL,
proof 13fc7c timed out internally at 300 seconds. It remains inconclusive.

New live handles (poll before restarting):
- 72199: FIR 16-tap 16/32-bit verify-only; 16-bit SAT timed out.
- 22304: dot product 8-lane 16/32-bit verify-only; 16-bit SAT timed out.
- 15640: saturating signed add/subtract 32/64-bit physical study; 32-bit SAT passed.
- 26439: barrel shifter 32/64-bit physical study, matched Basic flow.
- 62972: argmax 16 lanes, 16/32-bit physical study, matched Basic flow.

All failed/unknown results remain in the final audit. Remaining work includes
14 operations, three additional qualifying families, full family/config coverage,
missing proofs, ablations and the all-case geometric-mean gate. Published commit
a112017 passed hosted CI. Focused new datapath/sweep/registry tests: 64 passed.

Final checkpoint validation: 218 passed, six local-Yosys skips in the full focused
suite (`--basetemp=.ic/pytest-expanded-13`); make lint sim passed. FIR session
72199 is TERMINAL 18c448; dot session 22304 is TERMINAL fcc669. All four vector
checks passed, all four 120-second SAT attempts timed out. Their complete results
are retained as fir-verification.json and dot-product-verification.json. Physical
studies 15640, 26439 and 62972 remain active. No additional qualifying family yet.

## 39-operation checkpoint

Implemented CRC, LFSR jump and exact affine netlist equivalence. Operation 15
replaces the compressor alias already covered by operation 14. Eleven operations
remain. GF(2) proof controls passed: 4cffd4 positive, 40d646 wrong-polynomial
counterexample replayed in Icarus, nonlinear source rejected. Primary paper
connections and bounded proof contracts are in the modular READMEs.

Terminal studies:
- 15640 -> c3969a saturation: all pairs complete, but 64-bit LUT growth 28% exceeds
  the throughput-gain resource limit. Whole family does not qualify.
- 26439 -> 493114 barrel: all pairs complete, no material improvement.
- 27823 -> 1ca255 CRC verification: both affine proofs pass, case1 unrolled vector
  simulation hits 60 s; both SAT checks time out. Preserve the original record.
- 14821 -> 6455f0 LFSR verification: both affine/vector checks pass; SAT passes
  for 64-bit case and times out for 32-bit case.

Live physical handles: 23418 CRC (vector bound 300 s, same vector coverage),
98883 LFSR, and 62972 argmax pending authoritative terminal status. Poll before
restarting. Completed qualifiers remain priority, MCM, distributed FIFO until
another whole-family audit passes. New focused tests: 73 passed, six optional
local-Yosys skips; final full suite running as session 60342.

Final full focused validation: 259 passed, eight optional local-Yosys skips
(`--basetemp=.ic/pytest-expanded-14`). Real container affine controls passed
separately, including an Icarus-replayed witness. `make -C src/test lint sim`
passed. Previous published 083c3d6 CI passed; this checkpoint is not yet CI-verified.
Argmax is still live with two complete 32-bit pairs; do not prematurely count it.

## 42-operation checkpoint

Current inventory: 42. New operations 22/31/44 are iterative multiply, exact
divide and independent arithmetic cycle validation. Eight operations remain:
21 Booth, 35 banked regfile, 36 FSM, 37 skid, 38 systolic tile, 43 timing audit,
47 ablation and 50 whole-study acceptance audit. Architecture variants do not
count separately. Twenty-four substantial core/fixture cycle experiments pass.

Terminal physical studies:
- 62972 -> 0402f6 argmax: both sizes and all pairs qualify for throughput,
  +106.43%/+99.44%, LUT growth only 2.71%/2.75%. Fourth qualifying family.
- 98883 -> 2a7b3c LFSR: both measured, neither reaches the 15% gate.
- 23418 -> 2a0316 CRC: both measured, neither reaches the 15% gate.

Live sessions (poll before restarting): 41858 multiplier and 63223 divider.
Each has four predeclared cases, 16/32-bit folding and digit-size comparisons,
three paired runs, matched Basic flow and bounded retained attempts. Cycle
contracts may differ between architectures; every one is independently verified.
Fixed-width sequential simulation is not an unbounded formal proof.

Previous published 3e116d1 passed hosted CI. Local focused suite is in progress
as session 70872; repository lint/sim passed. Remaining usage 61%, above the
mandatory below-30% stop threshold. No reset credit consumed. Expanded acceptance
is incomplete; PR #81 stays draft and no merge is authorized.

Full focused validation: 290 passed, eight optional local-Yosys skips (the test
list in .github/workflows/ci.yml, --basetemp=.ic/pytest-expanded-15). All 24 real
arithmetic cycle experiments pass. make -C src/test lint sim passed; unchanged
hardware simulation targets are current. New-head hosted CI remains pending.
