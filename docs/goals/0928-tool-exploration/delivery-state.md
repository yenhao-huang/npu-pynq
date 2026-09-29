# GitHub delivery state

Current target: 50 tools; acceptance incomplete. The latest checkpoint below
replaces historical scope/status entries. Never treat the initial milestone as
whole-goal completion.

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
| Validate and commit | completed | 583 Python tests pass with 37 optional skips; lint passes; all eight RTL simulations pass |
| Draft PR | completed | Description has the current 55/59 result, nine qualifying families, evidence scope and limitations |
| Push, create and verify | completed | https://github.com/yenhao-huang/npu-pynq/pull/81; open draft, base dev, head npu/issue80-a |
| Handoff | completed | PR attached; goal, report, reproduction guide, skill state and evidence are committed |

Remaining usage at the latest check: 26%. The user's current instruction overrides
the default threshold for this run and stops at 20%. OpenSpec CLI is not installed.
Raw evidence is in .ic and compact summaries are under
docs/goals/0928-tool-exploration/evidence/. Consult the PR for the current head CI.

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

## Proof and timing integration checkpoint

All 50 planned operations exist; whole-study acceptance is still incomplete.
Bitwise SAT requires every output obligation. Product abstraction shares only
identical multiplication cells and proves every bit with arbitrary defined
products. An explicit total-binary-cell allowlist plus an original-netlist X/Z
check prevents hiding undefined behavior. Ten real SAT controls cover positive,
highest-bit mutation, different products, X outputs and division by zero.
Initial private-wire abstraction controls failed; public witness naming fixed
the exposure. Their failed run handles remain c87d20 and de8e14.

CSD retry fe89db proves 24/24 output bits at width16; width32 reaches only
23/48 before timeout and remains unknown. The full dot-product retry is pending.
Geometry, objectives, original failures and all denominator cases are preserved.

Completed prefix study 9418c2 has LUT growth 450%/718.75% and throughput losses
34.07%/20.19% at widths32/64. One-hot study bf8a99 has unchanged LUTs and throughput
losses 4.48%/1.79% at widths16/32. All 24 timing audits pass; neither family gains.
Matrix study bb9d6b completes all physical pairs but loses throughput
84.51%/82.91%. Its width16 LUT growth is 2200%; width32 LUT saving is 15.04%.
Width32 remains excluded from complete evidence because its baseline timing
coverage fails. Strict auditing is now automatic for coverage-bearing records;
historical records without coverage remain explicitly unaudited.

The current audit has 37/59 complete comparisons and seven qualifying families.
The all-case geometric benefit remains undefined. This is not a completion claim.
See acceptance-status.md and evidence/ for the exact current snapshot.

Validation for this checkpoint: the complete explicit CI tool-test list passed
448 tests with 15 optional native-Yosys skips (`--basetemp=.ic/pytest-expanded-21`).
Ten actual container controls pass. Repository `make -C src/test lint sim` passes
via MSYS2. A preceding focused run encountered one Windows run-store rename
PermissionError; the isolated retry passed (27 passed, 11 optional skips), and
the subsequent complete tool-test list passed. Published 7ac8e6d CI passed;
new-head hosted CI is pending. OpenSpec CLI remains unavailable.

Live sessions at this checkpoint: popcount physical 84635, CSD physical 69232,
dot-product proof 23888 and matrix DSP diagnostic 44357. Inspect these before
starting replacements. No goal completion, merge, reset credit or artifact
cleanup is authorized by this checkpoint.

## Source-normalization and DSP coverage checkpoint

There are still 50 operations and seven qualifying families. Complete comparisons
increase to 40/59; the all-case aggregate remains undefined. Three new completed
configurations retain their original objectives and all three paired repeats:

| Case | LUT reduction | Throughput change | Timing audits |
| --- | ---: | ---: | --- |
| CSD width16, constant255 | 76.00% | +102.60% | 6 pass |
| Modulo width16, modulus15 | 62.22% | +104.30% | 6 pass |
| Popcount width64 | -6.25% | -3.66% | 6 pass |

The larger CSD/modulo cases remain unproved, so neither is a new qualifying
family. Both dot-product abstraction retries time out at 10/35 and 10/67 bits.
No partial proof is promoted. Arithmetic normalization proves the original
128-bit popcount diagnostic; its formal-gated physical retry is running.

Every formal mode now checks each source before normalization/miter optimization
can erase partial behavior. Forty-five actual positive/negative controls match;
two earlier false-positive development controls are explicitly rejected and
preserved. Source widths, constants, objectives and historical failures remain.

The narrow DSP48E1 audit distinguishes unused ADREG/DREG defaults from active
registers using complete per-cell properties. Real bypassed/active-unclocked
controls respectively pass/fail. Matrix32 retry514661 accepts the baseline but
still rejects four candidate DSPs with CREG=1; do not generalize that exception.
Its PPA result is unchanged and remains a regression with incomplete coverage.

Windows run publication now retries only WinError5/32 for at most 1.26 seconds,
retaining the atomic rename and all evidence on permanent failure. The dedicated
13-test run-store suite passes and is included in CI. The complete explicit tool
list passes 495 tests with 17 native-Yosys skips; final source-guard controls and
focused tests run separately. Repository lint/sim passes. 636fbf3 hosted CI passes.
New commits retain draft PR #81; full acceptance and independent review remain.

Popcount width128 retry c3a4fc is now complete: all three pairs grow LUTs 2.18%
and lose throughput 9.21%; all six timing audits pass. The normalized proof
closes an evidence gap but creates no PPA win. Current audit11c679 has 41/59
complete comparisons. Only adder physical session18618 remains live.
Final source-safety regression: 40 passed, 28 optional native-Yosys skips;
45 actual container controls cover all skipped normalization safety cases.
Usage remaining at this checkpoint: 43%; stop below30%, no reset credit used.

## Complete semantic and provenance review checkpoint

All 50 operations have reviewed purposes, contracts, architecture boundaries and
failure-path evidence. The microarchitecture guide documents every baseline and
candidate. The current proof contract requires complete interfaces and defined
binary source semantics; 98 historical SAT interfaces pass source-bound
re-elaboration. FIR16/32 and Booth16/32 remain unproved after bounded attempts.

All 182 preserved pre-coverage Vivado records match their original run envelopes,
source bytes, raw reports, Tcl flows, tool identity and parsed metrics. All 57
parent study envelopes fix their objectives and full architecture payloads before
362 successful physical children start. The old in-memory runs still lack the
newer timing-coverage reports and saved routed checkpoints.

Acceptance run a16ee8 reports 55/59 complete comparisons, nine qualifying
families at both substantial configurations and an undefined all-case geometric
benefit. The complete Python suite passes 583 tests with 37 optional native-Yosys
skips. The required make lint/sim target passes; all eight simulations were also
forced directly through Icarus/VVP. Hosted CI passed commits e811a31 and 3ac2ecb;
current head status is tracked on PR #81, which remains draft. No merge or reset
credit is authorized. The user's active stop threshold is 20% remaining; the
latest check showed 26%.
