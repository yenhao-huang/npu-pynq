# Archived hardware experiments

Moved on 2026-09-11 at the user's request from `build/issue59/` to this directory.
The complete baseline, overlap, banked-only and optimized runs are preserved,
including source snapshots, commands, simulation metrics/logs, synthesis and
routing reports/checkpoints. Original logs retain their historical paths.

Additional prior outputs were moved here:
- `build/vivado/npu_pe_dsp_check/` -> `pe_dsp_check/`
- `src/test/build/` -> `regression/`
- `.Xil/` -> `vivado_session_cache/`

Published source/result record: commit aa38e1f, PR #60, and
`docs/exp/2026-09-11-hardware-scaling.md`. The original unoptimized baseline is
c1c643e. Current 16x16 K=256 cold transaction baseline is 8,482 cycles.
These generated archives are intentionally ignored by Git.
