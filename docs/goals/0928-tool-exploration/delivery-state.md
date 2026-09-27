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
