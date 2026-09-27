# Exploration state

Current target: 50 tools; acceptance incomplete. The latest checkpoint below
replaces historical scope/status entries. Never treat the initial milestone as
whole-goal completion.

Run ID: 2026-0928-issue80-a
Started: 2026-09-28 Asia/Taipei
Scope: ten paper-informed tools, eight PPA-related; no interactive questions
Issue / branch: #80 / npu/issue80-a, base dev 4a106811
Tool target / PPA target: 10 / 8
Experiment path: exp/tool-exploration/
Remaining usage / stop threshold: 75% / 30%

| Step | Status | Evidence |
| --- | --- | --- |
| Scope and papers | completed | goal.md, survey.md; primary source links |
| Design and implementation | completed | openspec/changes/add-paper-based-eda-exploration; ten registry operations |
| Tests and experiments | completed | 32 focused tests; 10 matching final evidence records; real Vivado retry passed |
| Report and reproduction | completed | report.md, reproduce.md, survey.md and evidence/01-10.json |
| PR handoff | completed | PR #81: https://github.com/yenhao-huang/npu-pynq/pull/81; base dev, head npu/issue80-a |

Next command: follow CI and review on PR #81.
Blockers: no new-tool blockers. Full-suite Windows limitations are recorded in report.md; OpenSpec CLI unavailable.

## Expanded run

Status: in_progress, superseding the completed initial ten-tool milestone.
Target: 50 tools / at least 40 PPA-related operations.
Acceptance: ../../../../../docs/goals/0928-tool-exploration/acceptance-50.md.
Next: clocked PPA, scalable correctness, substantive microarchitecture generators.
Remaining usage: 74%; stop below 30%. No new questions requested.

### Network and FIFO experiments

The expansion now has 23 implemented exploration operations. Use the
architecture_sweep pipeline for new combinational studies: it fingerprints
core source, inputs and measured tool versions, validates generated core and
wrapper artifacts, and rejects concurrent writers. Current network SAT coverage
is 11/12 substantial configurations; 128-bit popcount remains inconclusive.
The FIFO scoreboard checks 64/128-word storage variants against an independent
queue with protocol negative controls. FIFO physical coverage remains pending.
Do not treat a one-cycle minimum queue latency as a fixed latency.

Active physical study at this checkpoint: exec session 3537, priority encoder
64/128 bits with three paired repeats. Poll its authoritative status before
starting a duplicate. Old reduction and formal sessions are terminal; see
expanded delivery-state.md for retained errors and timeouts.

### Arithmetic and physical analysis

Inventory is 28 implemented exploration operations. Priority encoder completed
all three paired runs at both 64/128 bits and passes the LUT-area gate; preserve
its predeclared throughput objective for global aggregation. MCM and revised
leading-zero physical studies remain active (sessions 31080 and 23666).
Default/Basic optimization and implementation-thread metadata must match within
each comparison. Do not mix historical profiles silently.

The shared subprocess helper now owns a Windows Job Object / POSIX process group;
formal Docker runs also use an internal timeout. AIG normalization is optional
and recorded. It does not resolve the inconclusive 32-bit CSD case. Final focused
validation is 135 passed, four local-Yosys skips, with real container controls.

### Registered FIFO and netlist diagnostics

Inventory is 32 implemented operations. MCM completed both configurations and
all three paired repeats: LUT reductions 50.23% / 52.25%, throughput gains
35.48% / 33.71%. Together with priority encoder, two families qualify so far.
Leading-zero Basic study b652b3 ended with three measurement failures and one
below-threshold result; retain it in the eventual all-case audit.

Four digest-bound netlist operations have actual Yosys generic/xc7 experiments.
They report structural estimates, never routed timing or Vivado resource claims.
FIFO core and timing-fixture checks are independent. The fixture delays observed
signals by two cycles and is not an external ready/valid adapter. Record minimum
latency separately from fixed latency. Auto RAM inference can exchange LUTs for
BRAM and fail the unconditional improvement gate.

Active sessions: 69500 (auto FIFO) and 21483 (distributed revision), each two
substantial configurations with three physical pairs and at most two attempts.
Poll before restarting; all attempts and original auto cases must remain visible.
Validation: 154 passed, six optional local-Yosys skips; make lint sim passed.
Remaining usage at this checkpoint: 66%; stop below 30%. Eighteen operations,
four more qualifying families and full acceptance aggregation remain unfinished.

Mersenne reducer implementation raises inventory to 33; 17 operations remain.
Its integer oracle and canonical-zero mutation checks passed (57 focused tests).
Verify-only session 5034 is active. Do not count it as a qualifying PPA family.
Published netlist/FIFO checkpoint 64a6f98 passed hosted CI.

Modulo verification 5034 is terminal e2f022: 16-bit proved, 32-bit internal SAT
timeout. A 300-second retry runs as session 80622. Original flag/log discrepancy
is preserved and corrected in the checker. FIFO auto 69500 is terminal d0d9f1;
both cases fail unconditional resource gates because BRAM increases. Distributed
FIFO session 21483 remains active. Relevant validation: 70 passed, four skips.

### Signed datapath and FIFO checkpoint

Inventory: 36; fourteen operations remain. Distributed FIFO 59e508 qualifies at
both sizes and three pairs: area -96.52%/-97.12%, throughput +50.19%/+49.96%, no
DSP/BRAM growth. It is family three. Auto FIFO tradeoffs remain in the audit.
Modulo 300-second retry 13fc7c timed out; it did not prove 32-bit equivalence.
Live sessions: FIR 72199, dot 22304, saturation 15640, barrel 26439, argmax 62972.
Refer to delivery-state.md before restarting. FIR-window arithmetic does not
include sample history or a streaming protocol. New focused tests: 64 passed.

FIR 72199 (18c448) and dot 22304 (fcc669) are TERMINAL: all vectors passed,
all SAT attempts timed out. Keep them out of qualifying-win counts. Physical
sessions 15640/26439/62972 remain active. Full focused validation: 218 passed,
six local-Yosys skips; make lint sim passed. Remaining usage last checked: 65%.

### Linear proof and CRC/LFSR checkpoint

Inventory 39; eleven operations remain. Exact affine proof replaces the planned
compressor alias. It proves coefficient identity from actual Yosys netlists,
rejects unsupported logic and preserves the separate SAT cross-check outcome.
CRC initial case1 simulation timed out; 300-second revision retains 8192 random
vectors. LFSR both sizes are proven affine and simulated. Saturation and barrel
physical studies are terminal and do not qualify. Live: CRC 23418, LFSR 98883,
argmax 62972. See delivery-state.md for handles and failure evidence.

Full focused validation at this checkpoint: 259 passed, eight optional local
Yosys skips. Actual container affine controls pass separately; repository gates
pass. Remaining usage last checked: 64%, above the mandatory 30% stop threshold.

### Iterative checkpoint

Inventory 42; eight operations remain (21,35,36,37,38,43,47,50). Argmax 0402f6
qualifies at both widths/all pairs and is family four. LFSR 2a7b3c and CRC 2a0316
are terminal below-threshold physical studies; preserve them in the audit.
Twenty-four real arithmetic core/fixture cycle checks pass. Live studies:
41858 multiplier, 63223 divider, each four declared cases/three pairs. Do not
restart without checking those handles. Remaining usage last checked: 61%.

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
