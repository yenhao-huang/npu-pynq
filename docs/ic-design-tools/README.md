# IC design tools

Lint, simulation, waveform debugging, waveform viewing and synthesis, packaged
so an agent's RTL loop gives feedback in seconds instead of waiting minutes for
Vivado — and so none of that feedback costs more than a few kilobytes of
context.

Implements [`docs/plans/2026-0921-001-ic-design-tools-architecture.md`](../plans/2026-0921-001-ic-design-tools-architecture.md).

- [What you get](#what-you-get)
- [Install](#install)
- [Attaching the tools to an agent](#attaching-the-tools-to-an-agent)
  - [pi](#pi)
  - [Claude Code](#claude-code)
  - [Codex, and anything else with a shell](#codex-and-anything-else-with-a-shell)
- [Adding a tool](#adding-a-tool)
- [Removing a tool](#removing-a-tool)
- [Where results are kept](#where-results-are-kept)
- [The daemon](#the-daemon)
- [Limits worth knowing](#limits-worth-knowing)

## What you get

| Tool | What it answers | Typical cost |
| --- | --- | --- |
| `lint` | Is this RTL well-formed? Every diagnostic, in full. | ~1 s |
| `sim` | Does the testbench pass? Returns a verdict plus waveform and log **handles**. | seconds to minutes |
| `signals` | What is in this waveform? | ms |
| `first_mismatch` | At which cycle did reference and DUT diverge? | one pass over the trace |
| `value_at` | What were these signals at that cycle? | one pass |
| `value_range` | How did this signal get to that value? | one pass |
| `show_wave` | Open the waveform for a **person**, signals preselected. | opens a window |
| `synth` | LUTs, FFs, DSPs, and — in `full` mode — real timing. | 2 s (Yosys) / minutes (Vivado) |

The central rule: **nothing large is ever returned**. A simulation writes a
184 MB waveform and a 12 MB log to disk and hands back
`{"wave": "166d08/wave.fst", "log": "166d08/sim.log"}`. The agent then queries
that handle. `first_mismatch` is the payoff — it turns "scroll a waveform until
you spot the divergence" into one line of JSON.

```
Agent layer    Claude Code | Codex | pi | CI | a person
                          │
                   localhost HTTP  (optional)
                          │
daemon layer   ic-toold — run records, job lifecycle
                          │
ic_core        lint | sim | debug | view | synth
```

Each layer is allowed to know only about the one below it. `ic_core` does not
know an agent or a daemon exists, which is why every tool is also directly
usable by a person at a shell.

## Install

```bash
pip install -e tools/ic          # core + CLI
pip install -e "tools/ic[all]"   # plus daemon, MCP server
```

External programs, all optional except the default backend of whatever you use:

| Program | Used by | Notes |
| --- | --- | --- |
| `verilator` | `lint`, `sim` (default) | 5.x. `--binary --timing` runs the repository's SystemVerilog testbenches unmodified. |
| `iverilog` | `lint`, `sim` (`--backend icarus`) | Four-state; use it for X-propagation and reset bugs. |
| `fst2vcd` | `debug` (default) | Ships with GTKWave. |
| `gtkwave` | `show_wave` | GUI only. |
| `yosys` | `synth --mode estimate` | |
| `vivado` | `synth --mode full` | Self-hosted; not on CI runners. |

Check what is actually usable:

```bash
ic doctor
```

`{"ok": true}` means every default backend is present. Missing tools are
reported per backend rather than crashing, so a machine without Vivado still
gets working lint, sim, debug and estimates.

## Attaching the tools to an agent

For automatic dependency setup and a stdio MCP entry point distributed through
npm, see [the npm package guide](../../tools/ic/README.npm.md). Its launcher
downloads a pinned, checksum-verified OSS CAD Suite and installs locked Python
dependencies on first startup. A local npm tarball can be tested before registry
publication. The Python tool implementations and MCP schemas are shared with
the existing CLI and pi integration.

Every surface is generated from the same registry and ends in the same
`ic_core.dispatch`, so an agent gets identical results whichever it uses.
`tools/ic/tests/test_surfaces.py` asserts that equality rather than assuming it.

### pi

pi has no built-in MCP support — an explicit upstream design choice — so the
integration is an **extension** plus a **skill**. Both are already committed:

| File | Role |
| --- | --- |
| `.pi/extensions/ic-design-tools.ts` | Registers all eight tools as native pi tools |
| `.pi/settings.json` | Points pi's skill discovery at `.codex/skills` |
| `.codex/skills/custom/ic_design/ic-design-tools/SKILL.md` | Tells the model when and how to use them |

Nothing else is needed — pi auto-discovers `.pi/extensions/` once the project
is trusted. Verify without spending a token on a model:

```bash
node tools/ic/integrations/pi/verify.mjs
```

It loads the extension through pi's own SDK, prints the registered tool names,
calls `lint` end to end, and checks the skill is discoverable.

To load them explicitly instead of by discovery:

```bash
pi -e .pi/extensions/ic-design-tools.ts --skill .codex/skills
```

The extension shells out to `ic`. Set `IC_BIN` if it is not on `PATH`, and
`IC_DAEMON_URL` to route through a running daemon.

### Claude Code

`.mcp.json` in the repository root registers the MCP server:

```json
{
  "mcpServers": {
    "ic-design-tools": { "command": "ic-mcp", "args": [], "env": {} }
  }
}
```

The tool list and every JSON Schema are generated from the categories'
pydantic models, so the MCP surface cannot fall behind the tools.

The same skill works here — Claude Code discovers `.codex/skills` when
configured to, and the CLI path below always works regardless.

### Codex, and anything else with a shell

The CLI is the universal fallback and needs no integration at all:

```bash
ic lint --files src/hw/rtl/systolic_array/npu_pe.sv --top npu_pe
ic sim  --files src/hw/rtl/systolic_array/npu_pe.sv src/hw/tb/systolic_array/tb_npu_pe.sv --tb tb_npu_pe
ic first_mismatch --wave 166d08/wave.fst --ref expected --dut accumulator
```

`ic tools` prints the full catalogue with JSON schemas, which is enough for an
agent to drive the tools with no adapter whatsoever.

## Adding a tool

The registry is the single source of truth. Everything below is generated from
it and needs no edits:

| Generated | From |
| --- | --- |
| MCP tool list and schemas | the category's pydantic models |
| CLI subcommand and flags | the same models, reflected |
| HTTP endpoint | the same models, handed to FastAPI |
| pi tool registration | `ic tools`, i.e. the same models again |
| `meta.json` inputs | `In` fields plus backend name and version |
| Sync or job path | `Op.long_running` |

### A new backend for an existing category — one file

Say you want Verible for linting. Create `ic_core/tools/lint/verible.py`:

```python
from ...registry import backend
from . import Issue, LintIn, LintOut

@backend("lint", "verible", requires="verible-verilog-lint",
         version_cmd=["verible-verilog-lint", "--version"])
class VeribleLint:
    def lint(self, params: LintIn, ctx) -> LintOut:
        result = run_process([...], log_path=ctx.run.artifacts / "lint.log", cwd=ctx.cwd)
        return LintOut(..., backend="verible", backend_version=ctx.backend_version)
```

Then add it to the imports at the bottom of `ic_core/tools/lint/__init__.py`.
That is the whole change. Agents keep calling `lint(files, top)`; the new
backend is reachable as `lint(..., backend="verible")`, and becomes the default
by changing `default_backend` in the category.

### A new category — one folder

Say you want `fmt`. Create `ic_core/tools/fmt/`:

1. `__init__.py` defines `FmtIn`/`FmtOut` (pydantic models, **every field with
   a `description`** — that description is the only documentation an agent
   sees), then calls `register_category(Category(name="fmt", ops=[Op(...)], ...))`,
   then imports its backends.
2. `verible.py` (or whichever) implements a method named after each op.

`ic_core/tools/__init__.py` discovers the folder automatically. The daemon, the
CLI, the MCP server and the pi extension pick it up with no changes.

Two rules the tests enforce:

- **Op names are globally unique.** They double as MCP tool names and CLI
  subcommands.
- **Read-only, frequently-called ops set `records_run=False`.** A waveform
  query that created a run directory would bury the simulations that matter.

Add a test to `tools/ic/tests/`. `test_registry.py` will already check the new
category's schemas, descriptions and backend coverage.

## Removing a tool

Delete the backend file and its import line — or delete the category folder.
Nothing else refers to either by name.

To hide a tool from agents without deleting it, remove its `Op` from the
category's `ops` list; the backend code stays and the tests that exercise the
implementation keep passing.

To disable a tool for one agent only, use that agent's own mechanism: `--tools`
in pi, or removing the server from `.mcp.json` in Claude Code.

## Where results are kept

One invocation, one directory:

```
.ic/runs/2026-09-21/143052-a3f21c-sim/
    meta.json      what ran, whether it worked, how long it took   (KB, kept forever)
    out.json       the summary the agent received                  (KB, kept forever)
    artifacts/     wave.fst, sim.log, utilization.rpt              (MB-GB, retention)
    work/          Verilator obj_dir, Vivado project               (GB, reclaimed first)
```

Grouped by run rather than by tool, because the question asked afterwards is
almost always "what did that last run do", and one simulation's waveform and
log are one event.

A **handle** is `run_id/filename`. It maps straight to a path, so it stays
valid across daemon restarts. `meta.json` is the truth; `.ic/index.db` only
holds the queryable columns and can be deleted and rebuilt:

```bash
ic runs --top npu_systolic_array --limit 5
ic run a3f21c
ic artifact a3f21c/sim.log --grep FAIL
ic reindex
ic gc --artifacts-before 2026-09-01   # work/ always, old artifacts/ on request
```

Writes are atomic: a run is built in a `.tmp` directory and renamed on
completion, so an interrupted run can never be mistaken for a result.

Input files are recorded by SHA-256, not copied. There is **no caching** — the
same inputs run again produce a new run directory.

## The daemon

Optional. `lint`, `debug` and `view` never need it; `sim` and `synth` benefit
because a job survives the client disconnecting.

```bash
ic serve --port 8731          # or: ic-toold --port 8731
export IC_DAEMON_URL=http://localhost:8731
```

It does exactly two things — persist run records and manage job lifecycle —
and holds no tool knowledge at all.

Long-running ops submit a job and long-poll. A run that finishes inside the
window answers inline; a longer one returns `{"job_id": ...}`:

```bash
ic synth --files src/hw/rtl/... --top npu_matrix_accelerator --mode full
ic job 46daf65c --poll-s 240
```

Long polling rather than status polling is deliberate: a 40-minute synthesis
costs about eight calls, each of which carried information, instead of eighty
`{"status": "running"}` objects sitting in the agent's context.

Concurrency defaults to one job. Keep it there where Vivado is involved — a
single instance is several gigabytes.

## Limits worth knowing

- **Verilator is two-state.** It will not reproduce an X-propagation bug. For
  reset and initialisation problems use `--backend icarus`.
- **Tracing is injected, not edited in.** The testbenches do not call
  `$dumpfile`/`$dumpvars`; `sim` generates a one-module probe per run and
  `bind`s it into the testbench (Verilator) or elaborates it as a second
  top-level module (Icarus 11, which does not support `bind`). Either way the
  checked-in sources stay untouched.
- **FST is read by conversion.** There is no maintained pure-Python FST reader,
  so the `fst` backend expands the trace once with `fst2vcd` and caches the
  result under `.ic/cache/`. Simulations still write FST, so what is stored
  stays small.
- **A Yosys estimate has no timing.** It is technology-mapped but not placed or
  routed. Never report it as timing; use `--mode full`.
- **GTKWave is GUI-only.** `show_wave` hands a problem to a person; nothing can
  be read back from it.
