# Publication validation

Issue: #77. Branch: `npu/issue77-a`, based on `dev` at `8303261`.
All commands below completed on 2026-09-25 in the publication worktree.

- `make -C src/test lint sim` inside the existing helper container: PASS. The existing NPU lint and self-checking simulations completed; no NPU source was changed.
- `python exp/fp32_multipier/test/check_reference.py`: PASS, 2,018,272 independent nearest-grid reference checks.
- From the experiment directory, `python reproduce.py 034_onehot_fix 022_packed_priority 026_parallel_exp`: PASS. Each fresh RTL and gate-level simulation checked all 16,777,216 input pairs. Area/delay/ADP matched the frozen originals exactly.
- `python exp/fp32_multipier/audit.py`: PASS. Original logs, hashes, ranking, 32 successful evaluations, 22 distinct RTL snapshots and the three fresh replays agree.

Use the environment settings in `reproduce.md` for the host PDK path and helper mount. Fresh publication replay outputs remain ignored; original reviewed measurement fixtures are included under the narrow filetree exception.

No Vivado synthesis, routed timing, board execution or silicon power measurement was performed. The experiment's OpenROAD timing is post-mapping with zero wire parasitics. GitHub CI status is reported separately in the PR and is not inferred from these local checks.
