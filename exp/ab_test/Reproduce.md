# Reproduce the IC tools A/B experiment

Run commands from `exp/ab_test/` with Python 3.10 or newer. The published
[Report.md](Report.md) records the historical results. The attempt archive,
model CLI streams, and generated RTL snapshots are not in Git; reproducing
those exact numbers requires the original local `results/` archive. A new model
run is stochastic and will give different measurements.

## Environment

| Component | Version or location |
|---|---|
| Harness and grader | `src/` in this directory |
| IC tools | `npu/issue74-b` at `cdb514f` |
| Problems | RTLLM v2.0, pin `9d5f61a`; task inputs under `problems/` |
| PDK | Nangate45 Liberty and LEF under `~/pdk/` |
| Tools | Icarus Verilog 13.0, Yosys 0.69+post, Verilator 5.052, OpenROAD 26Q1-1744 |
| Model CLIs | Claude Code 2.1.288 and Codex CLI 0.160.0 for the historical run |

`setup/setup.sh` installs or checks the macOS arm64 environment, creates a
pinned IC-tool worktree, and writes an ignored `.ab_env` file. It requires
Homebrew, Docker Desktop, and authenticated model CLIs. Run:

```bash
bash setup/setup.sh
source .ab_env
python3 -m unittest discover -s tests
python3 -m src.main --list
```

The task descriptions are given to the model. References and testbenches are
used only by the grader. The historical setup also kept the upstream RTLLM
checkout outside the model workspace under `~/.ab_private/RTLLM`.

## Run the three study groups

```bash
analysis/run_all.sh   # Claude baseline, single-round, and multi-round
analysis/run_gpt.sh   # gpt-6-sol groups
```

The main run used 50 tasks. Baseline and single-round each used three
independent attempts per task and one round per attempt. Multi-round used one
attempt per task and five rounds. Claude had a $3.00 per-call budget; Codex CLI
had no budget flag. `analysis/rerun_multi.sh <model>` retries tasks without a
clean multi-round attempt. The report's shell comparison was a separate run:

```bash
analysis/run_shell.sh
```

The scripts write full output to the ignored `results/` directory. Review
`src/main.py` and the run scripts for the exact command-line arguments.

## Recalculate from a local run archive

The analysis programs default to `results/`. Set `AB_RESULTS` if your archive
is elsewhere. They write generated summaries into that directory:

```bash
python3 analysis/analyze.py
python3 analysis/extras.py
python3 analysis/report_tables.py
python3 analysis/make_tables.py
python3 analysis/multi_round_analysis.py
```

`analysis/verify_top3.py` and `analysis/regrade.py` require the archived RTL
snapshots and the pinned IC grading environment. The historical review
regraded all 745 selected main-study frontier snapshots and all 251 shell
baseline snapshots; their power, area, and delay measurements matched the
saved run data within relative error 1e-9. Those snapshots and verification
JSON files are not included in this compact pull request.

The historical multi-round token analysis also needs the CLI streams in the
local archive. The published report and `docs/multi-round-analysis.md` preserve
its findings, but this checkout alone cannot rerun that analysis.
