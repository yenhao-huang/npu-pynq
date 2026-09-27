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
