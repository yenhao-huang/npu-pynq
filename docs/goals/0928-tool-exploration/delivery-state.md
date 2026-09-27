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
