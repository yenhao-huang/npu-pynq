# Reproduce the top three FP12 designs

This work publishes the completed experiment in [issue #77](https://github.com/yenhao-huang/npu-pynq/issues/77), related to the original tracking issue #76. The directory spelling is preserved from the request; the format is FP12 E5M6.

## Prerequisites

Use a Windows host with Python 3.12, Docker Desktop, and an existing Linux helper container containing Verilator 5.006, Yosys 0.23 / ABC 1.01, g++ and make. The scripts require only the Python standard library. Original helper Python is 3.11.2; its experiment-local `.venv` is not committed. Tools are reused, not installed globally.

Pull the immutable official OpenROAD image:

```powershell
docker pull openroad/orfs@sha256:7f9c688e71cd49379b9e9a45a4558724e3eda501eaaaf11bc9b77382a3fe647b
```

The image reports version `unknown`; [environment.json](environment.json) records its executable hash. See the [official Docker documentation](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/master/docs/user/BuildWithDocker.md).

ASAP7 must already be available from OpenROAD-flow-scripts: four RVT TT NLDM combinational Liberty files and the original technology/cell LEFs. Original Liberty hashes are recorded in [docs/pdk-sources.json](docs/pdk-sources.json). No PDK, paper, compiled model, generated cell model or mapped netlist is vendored.

## Publication worktree commands

The example paths below match the tested local workspace. On another checkout, adjust the host location, `EVAL_LINUX_ROOT` (the same experiment directory as seen inside the helper) and `EVAL_PDK` (host PDK directory). `EVAL_CONTAINER` defaults to `codex-sandbox-agent-workspace`.

```powershell
Set-Location C:/Users/User/Desktop/agent_workspace/npu/worktrees/npu-issue77-a/exp/fp32_multipier
$env:EVAL_LINUX_ROOT='/workspace/npu/worktrees/npu-issue77-a/exp/fp32_multipier'
$env:EVAL_PDK='C:/Users/User/Desktop/agent_workspace/npu/npu_repo_in_pynq/external/OpenROAD-flow-scripts/flow/platforms/asap7'
python test/check_reference.py
python reproduce.py 034_onehot_fix 022_packed_priority 026_parallel_exp
python audit.py
```

Run these in a fresh checkout/output directory. `reproduce.py` refuses to overwrite an existing run. Use `--prefix audit` for another replay; `audit.py` checks the default `reproduce` prefix. Existing evidence is not deleted automatically.

Replay reads the original `runs/<name>/design.sv` bytes, independently of the current generator, and restores the original target delay and ABC settings. It:

1. Compiles RTL with Verilator and checks all 16,777,216 input pairs.
2. Runs Yosys/ABC mapping and `check -assert` against the configured Liberty.
3. Generates functional cell models from that exact library, compiles the mapped design, and checks all 16,777,216 pairs again.
4. Runs digest-pinned Docker OpenROAD with 10 ps input slew, 3.898 fF output load and zero wire parasitics.
5. Requires the RTL hash to match and area/delay/ADP to agree within 1 ppm; `audit.py` additionally checks exact observed agreement for this recorded campaign.

The generated `reproduce_verification.json`, replay logs, mapped netlists and acceptance output remain ignored. Checked-in source fixtures contain only the reviewed original source/metrics and concise synthesis/timing/verification logs under the narrow repository source-artifact exception.

## Expected results

| Rank | Original run | Area (um^2) | Delay (ps) | ADP (um^2 ps) |
|---|---|---:|---:|---:|
| 1 | 034_onehot_fix | 48.12858 | 863.236816 | 41546.362158 |
| 2 | 022_packed_priority | 47.15172 | 944.229614 | 44522.050375 |
| 3 | 026_parallel_exp | 50.18436 | 971.649963 | 48761.631537 |

All three use constrained ABC mapping with a 700 ps target. Their measured delays exceed 700 ps: the target is not a timing-closure claim. The source hashes and original settings are in [top3.json](top3.json).

## Evaluate new RTL

The top must be `fp12_mul`, with 12-bit a/b/y ports and the [numeric contract](docs/contract.md).

```powershell
python run_eval.py --name my_candidate --rtl path/to/design.sv --delay 700 --constrained --exhaustive
python run_eval.py --name onehot_new --mode onehot --mul operator --delay 700 --constrained --exhaustive
```

Each log starts with the full command. Failed runs are excluded from ranking. `python resume_sta.py <run>` can retry a terminal run whose RTL/gate checks passed; it verifies source/library hashes first. `python summarize.py` regenerates the report, top-three selection and local CSV after an intentional new campaign.

Power is a vectorless OpenSTA estimate, not a measured workload result. The FP16 paper comparison and all measurement limitations are explained in [report.md](report.md).
