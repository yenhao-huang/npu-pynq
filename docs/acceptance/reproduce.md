# Reproducing the IC design tools

Every tool, end to end, with the output each one actually produced. Runs on a
clean checkout with no board and no Vivado.

Verified 2026-09-21 on Debian 12 / WSL2 with Verilator 5.006, Yosys 0.23,
Icarus Verilog 11.0, GTKWave 3.3.118, Python 3.11, Node 20.20.2, pi 0.74.2.

- [0. Setup](#0-setup)
- [1. doctor — is the toolchain usable?](#1-doctor--is-the-toolchain-usable)
- [2. lint](#2-lint)
- [3. sim](#3-sim)
- [4. signals](#4-signals)
- [5. first_mismatch](#5-first_mismatch)
- [6. value_at](#6-value_at)
- [7. value_range](#7-value_range)
- [8. show_wave](#8-show_wave)
- [9. synth](#9-synth)
- [10. The daemon and the job path](#10-the-daemon-and-the-job-path)
- [11. The three surfaces agree](#11-the-three-surfaces-agree)
- [12. pi-agent](#12-pi-agent)
- [13. MCP](#13-mcp)
- [14. The test suite](#14-the-test-suite)
- [What was not verified here](#what-was-not-verified-here)

## 0. Setup

```bash
sudo apt-get install -y verilator yosys iverilog gtkwave zlib1g-dev
pip install -e "tools/ic[all]"
```

`zlib1g-dev` is not optional: Verilator compiles its FST writer from source,
and without zlib headers `--trace-fst` fails at the C++ stage with
`fatal error: zlib.h: No such file or directory`.

Run everything below from the repository root. To keep the run store out of
the working tree, export `IC_ROOT`:

```bash
export IC_ROOT=$(mktemp -d)
```

The fixture used throughout is `tools/ic/tests/fixtures/counter.sv`: a
reference counter and a copy with one deliberate bug — it skips a single
increment at 7. The two agree for seven cycles and then diverge by one
forever. Finding that by eye means scrolling a waveform, which is the problem
these tools exist to remove.

## 1. doctor — is the toolchain usable?

```bash
ic doctor
```

```json
{
  "ok": true,
  "ops": ["signals", "first_mismatch", "value_at", "value_range",
          "lint", "sim", "synth", "show_wave"],
  "backends": [
    {"category": "debug", "backend": "fst",       "default": true,  "available": true},
    {"category": "debug", "backend": "vcd",       "default": false, "available": true},
    {"category": "lint",  "backend": "verilator", "default": true,  "available": true},
    {"category": "lint",  "backend": "iverilog",  "default": false, "available": true},
    {"category": "sim",   "backend": "verilator", "default": true,  "available": true},
    {"category": "sim",   "backend": "icarus",    "default": false, "available": true},
    {"category": "synth", "backend": "yosys",     "default": true,  "available": true},
    {"category": "synth", "backend": "vivado",    "default": false, "available": false},
    {"category": "view",  "backend": "gtkwave",   "default": true,  "available": true}
  ]
}
```

`ok: true` with Vivado absent is the expected state on a machine without it:
only `synth --mode full` is unavailable, and it says so when asked (step 9).

## 2. lint

Clean RTL from the repository:

```bash
ic lint --files src/hw/rtl/systolic_array/npu_pe.sv --top npu_pe --compact
```

```json
{"ok": true, "error_count": 0, "warning_count": 0, "issues": [],
 "backend": "verilator", "backend_version": "Verilator 5.006 2023-01-22 rev (Debian 5.006-3)",
 "run_id": "8dc51a"}
```

A file with a real defect — note that warnings do not fail a lint, so check
`error_count`, not only `ok`:

```bash
ic lint --files tools/ic/tests/fixtures/counter.sv --top counter_ref
```

```json
{
  "ok": true, "error_count": 0, "warning_count": 1,
  "issues": [{
    "file": "tools/ic/tests/fixtures/counter.sv", "line": 8, "column": 8,
    "severity": "warning", "code": "DECLFILENAME",
    "message": "Filename 'counter' does not match MODULE name: 'counter_ref'"
  }]
}
```

The second backend, proving the category contract holds across tools — the
schema is identical, only `backend` changes:

```bash
ic lint --files tools/ic/tests/fixtures/tb_counter.sv --top tb_counter --backend iverilog --compact
```

```json
{"ok": false, "error_count": 2, "warning_count": 0,
 "issues": [
   {"file": "tools/ic/tests/fixtures/tb_counter.sv", "line": 9, "severity": "error",
    "message": "error: Unknown module type: counter_ref"},
   {"file": "tools/ic/tests/fixtures/tb_counter.sv", "line": 10, "severity": "error",
    "message": "error: Unknown module type: counter_broken"}]}
```

## 3. sim

```bash
ic sim --files tools/ic/tests/fixtures/counter.sv tools/ic/tests/fixtures/tb_counter.sv \
       --tb tb_counter
```

```json
{
  "ok": false, "built": true, "exit_code": 0, "timed_out": false,
  "duration_s": 5.933, "pass_count": 0, "fail_count": 1,
  "summary": ["FAIL tb_counter: 14 mismatching cycles", "..."],
  "summary_truncated": true,
  "wave": "c29f5d/wave.fst",
  "log": "c29f5d/sim.log",
  "backend": "verilator",
  "run_id": "c29f5d"
}
```

The verdict and the lines that explain it come back; the waveform and log stay
on disk behind handles. Substitute your own `run_id` for `c29f5d` below.

Two things worth noting. Verilator ran the repository's SystemVerilog
testbench **unmodified** — `--binary --timing` accepts the delay and event
constructs it uses. And the testbench never calls `$dumpfile`/`$dumpvars`; the
waveform exists because `sim` generates a probe module and `bind`s it into the
testbench per run, leaving the checked-in sources untouched.

To confirm nothing large crossed back:

```bash
ic artifact c29f5d/sim.log --tail 3
```

## 4. signals: List singals

```bash
ic signals --wave <simulator-run-id>/wave.fst --pattern '*count*' --compact
```

```json
{"scopes": ["TOP", "TOP.tb_counter", "TOP.tb_counter.u_dut", "TOP.tb_counter.u_ref"],
 "signals": [{"path": "TOP.tb_counter.ref_count", "width": 8, "kind": "logic"},
             {"path": "TOP.tb_counter.dut_count", "width": 8, "kind": "logic"}, "..."],
 "total": 14, "truncated": false, "timescale": "1ps",
 "clock": "TOP.tb_counter.clk"}
```

The clock was detected without being named, which is what makes every cycle
argument below work.

## 5. first_mismatch

The tool the rest exists to support.

```bash
ic first_mismatch --wave <simulator-run-id>/wave.fst --ref ref_count --dut dut_count --context 3
```

```json
{
  "found": true, "time": 85000, "cycle": 8,
  "ref": "TOP.tb_counter.ref_count", "dut": "TOP.tb_counter.dut_count",
  "ref_value": {"bin": "00001000", "hex": "0x08", "dec": 8, "signed": 8, "unknown": false},
  "dut_value": {"bin": "00000111", "hex": "0x07", "dec": 7, "signed": 7, "unknown": false},
  "context": [
    {"cycle": 5, "ref": {"dec": 5}, "dut": {"dec": 5}},
    {"cycle": 6, "ref": {"dec": 6}, "dut": {"dec": 6}},
    {"cycle": 7, "ref": {"dec": 7}, "dut": {"dec": 7}},
    {"cycle": 8, "ref": {"dec": 8}, "dut": {"dec": 7}}
  ],
  "compared_cycles": 9
}
```

This is exactly the injected bug: agreement through cycle 7, divergence at
cycle 8, the DUT stuck one behind. Note that the names passed were `ref_count`
and `dut_count` — a unique suffix resolves to the full hierarchical path.

Signals that agree are reported honestly rather than by silence:

```bash
ic first_mismatch --wave <simulator-run-id>/wave.fst --ref ref_count --dut ref_count --compact
# {"found": false, ..., "compared_cycles": 21}
```

## 6. value_at

```bash
ic value_at --wave <simulator-run-id>/wave.fst --signals ref_count dut_count enable --cycle 8 --compact
```

```json
{"time": 85000, "cycle": 8,
 "values": {"ref_count": {"hex": "0x08", "dec": 8, "signed": 8, "unknown": false},
            "dut_count": {"hex": "0x07", "dec": 7, "signed": 7, "unknown": false},
            "enable":    {"hex": "0x1",  "dec": 1, "signed": 1, "unknown": false}},
 "missing": []}
```

A name that matches nothing lands in `missing` rather than failing the call, so
one typo does not discard the other signals you asked for.

## 7. value_range

```bash
ic value_range --wave <simulator-run-id>/wave.fst --signal dut_count --from-cycle 5 --to-cycle 12 --max-points 5
```

```json
{"signal": "TOP.tb_counter.dut_count", "width": 8,
 "points": [{"time": 55000, "cycle": 5, "value": {"dec": 5}},
            {"time": 65000, "cycle": 6, "value": {"dec": 6}},
            {"time": 75000, "cycle": 7, "value": {"dec": 7}}],
 "total": 3, "truncated": false}
```

Three transitions across eight cycles, because the DUT stops counting at 7 —
the bug seen from the other side.

## 8. show_wave

For a person, not for an agent. On a headless machine, generate the save file
and stop there:

```bash
ic show_wave --wave <simulator-run-id>/wave.fst --signals ref_count dut_count enable \
             --center-cycle 8 --no-launch
```

```json
{"savefile": ".../083226-2365db-view/artifacts/view.gtkw",
 "command": "gtkwave .../wave.fst .../view.gtkw",
 "launched": false, "reason": "launch=false; save file only",
 "signals": ["TOP.tb_counter.ref_count", "TOP.tb_counter.dut_count", "TOP.tb_counter.enable"],
 "center_time": 85000}
```

The generated `.gtkw`:

```
[*] Generated by ic show_wave
[dumpfile] ".../wave.fst"
[treeopen] TOP.tb_counter.
[timestart] 0
*-16.0 85000 -1 -1 -1 -1 -1 -1
@22
TOP.tb_counter.ref_count[7:0]
@22
TOP.tb_counter.dut_count[7:0]
@28
TOP.tb_counter.enable
```

The `@22`/`@28` flag lines and the `[7:0]` bit ranges are load-bearing.
GTKWave silently ignores a bare signal name, so a save file without them opens
an empty window that looks like the tool worked.

To verify the GUI path with no monitor, use a virtual display:

```bash
sudo apt-get install -y xvfb
Xvfb :99 -screen 0 1280x800x24 &
DISPLAY=:99 ic show_wave --wave c29f5d/wave.fst --signals ref_count dut_count enable --center-cycle 8
# {"launched": true, "reason": null, ...}
pgrep -a gtkwave
```

![GTKWave opened by `ic show_wave`, with ref_count, dut_count and enable preloaded and the marker on cycle 8](../assets/ic-show-wave-first-mismatch.png)

The window opens on the divergence: `ref_count=08` against `dut_count=07`,
with the marker at 85 ns and `dut_count` visibly flat from `07` onward while
`ref_count` keeps counting.

## 9. synth

```bash
ic synth --files tools/ic/tests/fixtures/counter.sv --top counter_ref --mode estimate
```

```json
{
  "ok": true, "mode": "estimate", "part": null,
  "utilization": {"luts": 9, "ffs": 8, "dsps": 0, "brams": 0, "cells": 31, "memory_bits": 0},
  "timing": {"wns_ns": null, "tns_ns": null, "met": null},
  "report": "…/synth.log", "duration_s": 2.1, "backend": "yosys",
  "note": "Estimate only: technology-mapped but not placed or routed, and no timing analysis. Use mode='full' before trusting area or timing."
}
```

Eight flip-flops for an 8-bit counter, in two seconds. `timing.met` is `null`,
not `true`: Yosys does no timing analysis, and the tool says so rather than
letting a number look authoritative.

`mode="full"` selects Vivado from the mode alone. Without Vivado installed it
degrades to a clear, coded error rather than a crash:

```bash
ic synth --files tools/ic/tests/fixtures/counter.sv --top counter_ref --mode full --compact
```

```json
{"error": {"code": "backend_unavailable",
           "message": "synth backend 'vivado' needs 'vivado' on PATH",
           "details": {"backend": "vivado", "requires": "vivado"}}}
```

On a machine with Vivado this returns real utilization and a `wns_ns` for
`xc7z020clg400-1`, through the job path in step 10.

## 10. The daemon and the job path

```bash
ic serve --port 8731 &
curl -s localhost:8731/v1/health
```

```json
{"ok": true, "root": "…", "store": "…/.ic/runs",
 "ops": ["signals", "first_mismatch", "value_at", "value_range", "lint", "sim", "synth", "show_wave"]}
```

Submit without waiting, then long poll — the path a 40-minute Vivado run takes:

```bash
curl -s -X POST "localhost:8731/v1/tools/sim?wait_s=0" -H 'content-type: application/json' \
  -d '{"files":["tools/ic/tests/fixtures/counter.sv","tools/ic/tests/fixtures/tb_counter.sv"],"tb":"tb_counter"}'
# {"job_id": "46daf65c", "state": "queued", "op": "sim"}

curl -s "localhost:8731/v1/jobs/46daf65c?wait_s=120"
# {"job_id": "46daf65c", "state": "succeeded", "run_id": "1ebe15", "result": {...}}
```

A run that finishes inside the window answers inline instead, and the caller
never learns a job existed:

```bash
curl -s -X POST "localhost:8731/v1/tools/sim?wait_s=180" -H 'content-type: application/json' -d '{...}'
# {"ok": false, ..., "wave": "599e98/wave.fst", "run_id": "599e98", "job_id": "d6c553df"}
```

Run history and bounded artifact reads:

```bash
ic runs --limit 4
ic run c29f5d
ic artifact c29f5d/sim.log --grep FAIL
ic reindex          # index.db is disposable; meta.json is the truth
ic gc               # reclaims work/, keeps meta.json and artifacts/
```

## 11. The three surfaces agree

The claim the architecture rests on. Same request, three ways:

```bash
ic lint --files src/hw/rtl/systolic_array/npu_pe.sv --top npu_pe --compact
IC_DAEMON_URL=http://localhost:8731 ic lint --files src/hw/rtl/systolic_array/npu_pe.sv --top npu_pe --compact
curl -s -X POST localhost:8731/v1/tools/lint -H 'content-type: application/json' \
  -d '{"files":["src/hw/rtl/systolic_array/npu_pe.sv"],"top":"npu_pe"}'
```

All three return the same object apart from `run_id`.
`tools/ic/tests/test_surfaces.py` asserts this rather than leaving it to
inspection.

## 12. pi-agent

pi has no built-in MCP support, so the integration is an extension plus a
skill, both committed. Nothing needs configuring:

```bash
npm install -g @earendil-works/pi-coding-agent
node tools/ic/integrations/pi/verify.mjs
```

```
REGISTERED: ["first_mismatch","lint","show_wave","signals","sim","synth","value_at","value_range"]
lint schema required: ["files"]
lint result: {"ok": true, "error_count": 0, "warning_count": 0, "issues": [], "backend": "verilator", ...}
SKILLS: [..., "ic-design-tools", ...]
tools=true lint=true skill=true
```

Exit status 0. The verifier loads the extension through pi's own SDK, so the
tool list it prints is the list a model would see; it then calls `lint`
through the full path — pi → extension → `ic` → dispatch — and checks that the
skill is discoverable through the `skills` entry in `.pi/settings.json`. No
model and no API key are involved, so it runs in CI.

Interactively, once a provider is configured:

```bash
pi          # tools are auto-discovered from .pi/extensions/
pi -e .pi/extensions/ic-design-tools.ts --skill .codex/skills   # or load explicitly
/ic         # the extension's command: prints backend availability
```

## 13. MCP

For Claude Code, `.mcp.json` registers `ic-mcp`. To drive it as a client would:

```bash
python3 - <<'PY'
import asyncio, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "ic_mcp.server"], env=dict(os.environ))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print([t.name for t in tools.tools])
            print((await s.call_tool("lint", {"files": ["src/hw/rtl/systolic_array/npu_pe.sv"], "top": "npu_pe"})).content[0].text)

asyncio.run(main())
PY
```

```
['signals', 'first_mismatch', 'value_at', 'value_range', 'lint', 'sim', 'synth', 'show_wave']
{"ok": true, "error_count": 0, ...}
```

Both the mcp 2.x (`MCPServer`) and 1.x (low-level `Server`) APIs are
supported, because the SDK version an agent harness pins is not ours to
choose.

## 14. The test suite

```bash
python3 -m pytest tools/ic/tests -q
```

```
41 passed
```

Tests whose backend is missing skip rather than fail, so the suite is useful
on a machine with only part of the toolchain — which is the normal case, since
Vivado is on almost none of them.

## What was not verified here

- **`synth --mode full`.** Vivado is not installed on this machine. The
  backend-selection logic, the unavailable-backend error and the job path it
  uses are all covered; the Vivado invocation and its report parsing are not.
  Run it on the self-hosted synthesis machine before trusting timing.
- **The board.** Nothing here touches the PYNQ-Z1. These tools stop at
  synthesis.
- **Large traces.** The fixtures are kilobytes. The queries stream and hold
  flat memory by construction, and `first_mismatch` keeps only its context
  window, but a multi-gigabyte FST has not been timed.
- **pi with a live model.** Tool registration, schema and execution are
  verified through pi's SDK; whether a given model chooses to call them is a
  prompting question, which is what the skill in
  `.codex/skills/custom/ic_design/ic-design-tools/` addresses.
