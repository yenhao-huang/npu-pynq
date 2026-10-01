# Reproduce the FP16 adder experiment

Run commands from `exp/fp16_adder/`. The existing helper container must contain
Verilator and Yosys and mount the experiment at `EVAL_LINUX_ROOT`. Supply an
ASAP7 platform directory with the four RVT TT NLDM libraries and LEFs through
`EVAL_PDK`. No Python third-party packages are needed.

The recorded Windows worktree configuration is:

```powershell
$env:EVAL_LINUX_ROOT='/workspace/npu/worktrees/npu-issue77-a/exp/fp16_adder'
$env:EVAL_PDK='C:/Users/User/Desktop/agent_workspace/npu/npu_repo_in_pynq/external/OpenROAD-flow-scripts/flow/platforms/asap7'
$env:EVAL_CONTAINER='codex-sandbox-agent-workspace'
docker pull openroad/orfs@sha256:7f9c688e71cd49379b9e9a45a4558724e3eda501eaaaf11bc9b77382a3fe647b
python test/check_reference.py
python summarize.py
python reproduce.py
python audit.py --require-complete
```

`top3.json` selects three distinct RTL byte hashes ranked by mapped area times
delay. `reproduce.py` uses the frozen `runs/<name>/design.sv`, original mapping
settings and the pinned image. Each fresh `runs/replay_*` directory contains
commands, RTL and gate checks, synthesis and STA evidence. It fails on a
functional failure or any difference in source/library hashes, check counts,
area, delay or ADP. `replay_verification.json` records successful agreement.

The normal verification suite covers 3,359,296 pairs, not the full 32-bit input
space. Optional `run_eval.py --exhaustive` verifies 4,294,967,296 pairs per stage
and can take substantially longer. Do not describe normal replay as exhaustive.

To regenerate the campaign in a fresh experiment directory, run the baseline
command from `docs/contract.md`, then `python campaign.py`. Existing run
folders are immutable; choose a new run name for retries. Do not clear build
artifacts or overwrite historical measurement records to manufacture agreement.

## Selected designs

| Joint rank | Frozen design | Area (um^2) | Delay (ps) |
| --- | --- | ---: | ---: |
| 1 | 021_sentinel_priority | 51.99228 | 1094.322144 |
| 2 | 025_prefix_priority | 54.25218 | 1159.015625 |
| 3 | 027_round_sentinel | 52.05060 | 1208.955200 |

The selection is three distinct RTL snapshots, not repeated mappings of one
source. Fresh replay is checked by `audit.py --require-complete`.
