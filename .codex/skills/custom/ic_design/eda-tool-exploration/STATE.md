# Exploration state

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
