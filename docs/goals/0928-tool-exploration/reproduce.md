# Reproducing the exploration

Run from the issue worktree's repository root. Python 3.12 was used; package
compatibility is declared in tools/ic/pyproject.toml. Create a local environment:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e "tools/ic[all]" pytest pyyaml
```

For a pinned runtime baseline use tools/ic/npm/requirements.lock; tests also need
pytest and skill validation needs PyYAML. Keep the actual package versions with
your results. Do not install globally.

## Tool prerequisites

- Icarus Verilog 13.0 (`iverilog` and `vvp`) for exhaustive combinational testing.
- Local licensed Vivado 2026.1 for the recorded PPA experiment, targeting
  xc7z020clg400-1. Other versions require fresh measurements of both designs.
- GNU Make and Verilator for the repository gate; this Windows host uses MSYS2.
- No board, model weights, LLM API, downloaded datasets or external service.

`python -m ic_cli.main doctor` reports available backends. Missing Yosys,
fst2vcd or GTKWave does not prevent these new operations; the existing category
defaults may still make doctor return nonzero. Do not interpret that as every
operation being unavailable. Vivado must run under the account that owns its
working configuration; a restricted sandbox account may fail to load features.

## Run all ten experiments

```powershell
.venv/Scripts/python.exe exp/tool-exploration/run.py --tool all --resume
```

Run one tool with `--tool 01` through `--tool 10`. The matching config is in
`exp/tool-exploration/exp-tool-<id>-<paper>/config.json`. Results default to the
ignored `exp/tool-exploration/output/<id>.json`. Use a new `--output` directory
for an independent run. `--resume` checks the code/config/fixture and tool-version fingerprint;
without it a fresh measurement is performed. Changed tool versions invalidate the
cache. For other environment changes, choose a new output directory.

Tools 05-08 depend on tool 04 and will measure both designs if matching completed
measurements are unavailable. In one invocation completed dependencies are reused.
Tool 10 runs its own correctness and measurement sequence, plus a negative control.
A failed process or invalid measurement exits nonzero without writing a completed
experiment. Inspect its `.ic/` run record to diagnose, then rerun that tool.

## Expected behavior

The reference computes `sel ? (a+b) : (a+c)` and the candidate computes
`a + (sel ? b : c)` with four-bit inputs and modular output. Tool 03 checks all
8192 input combinations. The negative control subtracts and must fail. Tool 10
must stop that negative control at correctness before launching Vivado.

Tool 04 routes both designs under a common 10 ns input/output datapath constraint.
It records LUTs, FFs and worst input-to-output datapath delay. Power is absent.
Routed delay can vary with the tool version and implementation heuristics; judge
your run by its measured records, not by an assumed universal percentage.
Tool 06 reports improvement as `100 * (baseline - candidate) / baseline`;
a zero baseline returns a null percentage. Tool 07 retains ties and trade-offs.
Tool 08 refuses ineligible rewards; tool 09 cannot select an unaffordable action.

## Access the tools directly

```powershell
.venv/Scripts/python.exe -m ic_cli.main tools --compact
.venv/Scripts/python.exe -m ic_cli.main optimization_rules --topics area arithmetic
.venv/Scripts/python.exe -m ic_cli.main width_advice --a-min 0 --a-max 15 --b-min 0 --b-max 15 --operation add --declared-width 16
```

Complex CLI fields accept JSON objects (for baseline/candidate/weights/limits)
or one JSON object per list item (for ports, records, candidates). The experiment
runner uses the same dispatch/schema path as CLI, MCP and daemon. MCP and HTTP
clients send native JSON objects. Operation-specific backend defaults are in the
input schema; leave backend unset unless you intentionally choose an implementation.

Read detailed transcripts through handles, for example:

```powershell
.venv/Scripts/python.exe -m ic_cli.main run RUN_ID
.venv/Scripts/python.exe -m ic_cli.main artifact RUN_ID/measure.log --tail 30
```

Do not read waveforms or simulator logs directly. Evidence JSON contains compact
results and handles; raw artifacts remain machine-local in `.ic/`.

## Validation

```powershell
New-Item -ItemType Directory -Force .ic | Out-Null
.venv/Scripts/python.exe -m pytest tools/ic/tests/test_exploration.py tools/ic/tests/test_registry.py tools/ic/tests/test_cli_catalogue.py -q --basetemp=.ic/pytest-new-run
.venv/Scripts/python.exe -m pytest tools/ic/tests -q --basetemp=.ic/pytest-full-new-run
```

On this Windows host, run the repository gate with MSYS2 paths:

```powershell
C:/msys64/usr/bin/bash.exe -c 'export PATH=/ucrt64/bin:/usr/bin:$PATH; cd /c/Users/User/Desktop/agent_workspace/npu/worktrees/npu-issue80-a && make -C src/test lint sim'
```

Adjust only the worktree path for another checkout. Use unique pytest base-temp
folders to avoid cross-account temporary-directory permissions. See report.md
for actual results and baseline-reproduced Windows limitations.

## Workflow reuse and usage stop

Invoke `.codex/skills/custom/ic_design/eda-tool-exploration/SKILL.md` for another
survey. It asks for tool count/topic priorities and experiment paths only when
those are missing and questions are allowed. Check remaining Codex usage before
expensive batches, save state and stop below 30%. This rule concerns the agent
workflow; the standalone Python experiment runner has no account-usage API.

## Expanded network and FIFO studies

From the issue #80 checkout:

```powershell
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool all --container codex-sandbox-agent-workspace --verify-only
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 24 --container codex-sandbox-agent-workspace
.venv/Scripts/python.exe exp/tool-exploration/exp-tool-34-scalar/study.py
```

The container is an existing local Yosys provider, not created by these commands.
Omit --container when Yosys is installed locally. network_study uses per-tool
config.json declarations and stores each parent result under its own ignored
output directory. architecture_sweep checkpoints all predeclared cases beneath
.ic/studies using the complete input, core-source hash and measured tool versions.
An OS lock rejects duplicate writers to a study key. No raw logs are committed.

A positive SAT status is required before physical measurement by default. A
failed proof, missing source artifact or altered generated wrapper does not
become a cached success. verify-only changes the study identity deliberately.
The current physical example is priority encoder only; other families still
need matched physical runs. FIFO studies currently exercise protocol correctness;
whole-FIFO timing awaits a registered fixture. The old reduction runner predates
this stronger resume contract and must not be treated as equivalent provenance.

## Arithmetic and pair-analysis reproduction

```powershell
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 16 --container codex-sandbox-agent-workspace --verify-only
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 17 --container codex-sandbox-agent-workspace --verify-only
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 18 --container codex-sandbox-agent-workspace
.venv/Scripts/python.exe exp/tool-exploration/physical_analysis_study.py
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 25 --configuration config-binary-search.json --container codex-sandbox-agent-workspace
```

The last command uses the explicitly declared Basic optimization profile and
retains both original-tree and new binary-search candidates. Default and Basic
records are incompatible; implementation-thread limits are also compared.
Historical records without these metadata fields can be compared only to the
same historical profile, not silently merged with new records.

The preserved priority evidence contains the complete source-bound correctness
and physical parent result. physical_analysis_study rechecks its resource and
repeat gates without launching Vivado. It does not declare global acceptance.

Formal normalization defaults to word-level Yosys SAT. The optional aig mode
uses techmap/ABC before SAT and is recorded in the result. It did not resolve
the difficult 32-bit CSD case. Docker runs use an internal total time guard,
including ABC preprocessing; exit 124 is recorded as a timeout, never proof.

## Netlist and registered FIFO studies

```powershell
.venv/Scripts/python.exe exp/tool-exploration/netlist_study.py --container codex-sandbox-agent-workspace
.venv/Scripts/python.exe exp/tool-exploration/exp-tool-34-scalar/physical_study.py
.venv/Scripts/python.exe exp/tool-exploration/exp-tool-34-scalar/physical_study.py --distributed-revision
```

The two physical commands are separate predeclared candidate sets. Preserve both
results; distributed storage does not replace or erase auto-inference data.
The generic/xc7 netlists are estimates, and the RAM policy must be rechecked in
actual Vivado utilization. Core and timing-fixture scoreboards both gate each
physical case. The fixture delays observations by two cycles and is explicitly
not a ready/valid adapter for external integration. Minimum latency and fixed
latency are distinct measurement contracts.

architecture_sweep now records each bounded implementation attempt, including
failed attempts before a successful retry. Default maximum is two; this is not
permission to rerun indefinitely or discard failed results. Existing live studies
retain the implementation loaded at their start; do not launch duplicates merely
because newer source gives a new checkpoint key.
