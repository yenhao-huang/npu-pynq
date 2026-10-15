# Publication validation

Validated in `worktrees/npu-issue77-a` on 2026-09-27.

- `python exp/fp16_adder/test/check_reference.py`: PASS, 851,968 independent
  checks against binary64 addition followed by binary16 packing.
- Original campaigns: 28 passing evaluations, 20 distinct RTL snapshots,
  two retained compile failures. Every passing evaluation verifies 3,359,296
  pairs in RTL and again in the mapped gate model.
- From the experiment directory, `python reproduce.py`: PASS for
  `021_sentinel_priority`, `025_prefix_priority`, `027_round_sentinel`.
  All source/library hashes, check counts, area, delay and ADP match exactly.
- `python exp/fp16_adder/audit.py --require-complete`: PASS; checks original
  logs and frozen RTL hashes, ranking, >10 evaluations, and fresh replay data.
- `docker exec -w /workspace/npu/worktrees/npu-issue77-a
  codex-sandbox-agent-workspace make -B -C src/test lint sim`: PASS, lint and
  all eight repository testbenches; forced recompilation and execution.
- Python syntax compilation passed for all 14 authored Python files.

Fresh replay records:

| Original | Fresh run | Result |
| --- | --- | --- |
| 021_sentinel_priority | replay_20260927_153134_021_sentinel_priority | exact agreement |
| 025_prefix_priority | replay_20260927_153202_025_prefix_priority | exact agreement |
| 027_round_sentinel | replay_20260927_153244_027_round_sentinel | exact agreement |

Replay build products remain local/ignored. The report records the result and
`reproduce.py` plus `audit.py --require-complete` regenerate and verify it.

The best joint design is 51.99228 um^2 and 1094.322144 ps, approximately 6.1%
and 4.9% above the separate paper minima. This is a close PPA comparison with
explicit tool/constraint limitations, not a claim of identical paper flow or
1000 ps timing closure. No Vivado, routing, board or silicon test was run:
this is a standalone ASIC-library combinational experiment. Full 2^32-pair
exhaustive FP16 verification was not run; sampled coverage is stated exactly.
