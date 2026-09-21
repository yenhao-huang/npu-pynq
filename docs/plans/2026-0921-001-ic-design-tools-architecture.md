# IC Design Tools architecture

| Plan ID | Topic | Revision | Status |
|---|---|---|---|
| 2026-0921-001 | ic-design-tools-architecture | rev.4 (2026-09-21) | Implemented — see [5. Implementation results](#5-implementation-results) |

Goal: for this FPGA project (PYNQ-Z1 / Zynq-7020), bring the agent's RTL
iteration loop down from "wait several minutes for Vivado each time" to
second-scale feedback, while keeping the context budget under control. Single
machine. No ASIC flow, no compute farm, no multi-tenancy.

---

## 1. Layers

```
Agent layer    Claude | GPT/Codex | pi-agent | CI | a person
                          │
                   localhost HTTP
                          │
daemon layer   ic-toold — persists run records, manages job lifecycle
                          │
ic-core layer  Lint | Simulator | Debug | GTKWave | Synthesis
```

| Layer | Responsibility | What it does not do |
|---|---|---|
| Agent | A thin interface. The MCP server and the CLI are both just daemon clients | Holds no logic |
| daemon | Exactly two things: **persist run records** and **job lifecycle** | Holds no tool knowledge |
| ic-core | One module per tool, no agent dependency, independently testable and directly usable by a person | Does not know an agent or a daemon exists |

**Transport**: localhost HTTP (FastAPI + uvicorn). The MCP server and the CLI
call the same endpoints, so their results are the same.

**Core principle**: nothing returned to an agent exceeds a few KB. Large files
(waveforms, logs, reports) stay with the daemon; only a handle comes back.

---

## 2. Tool layer

### 2.1 Directory structure: the category defines the contract, the backend implements it

```
ic_core/tools/
  lint/     __init__.py   ← LintIn / LintOut / LintBackend contract
            verilator.py
            verible.py     (future)
  sim/      __init__.py   ← SimIn / SimOut / SimBackend
            verilator.py
            xsim.py        (future, for testbenches Verilator cannot take)
  debug/    __init__.py   ← WaveQuery contract
            fst.py
            vcd.py         (fallback)
  view/     __init__.py   ← Viewer contract
            gtkwave.py
  synth/    __init__.py   ← SynthIn / SynthOut
            yosys.py       (estimate)
            vivado.py      (full)
  registry.py
```

**The schema belongs to the category, not to the tool.** `lint()`'s input and
output are defined by `lint/__init__.py`; `verilator.py` is responsible only
for translating Verilator's output into `LintOut`. Swapping or adding a backend
leaves the agent side and the prompts completely untouched.

**An agent does not need to know which backend is behind a call**:

```
lint(files, top)                    # verilator by default
lint(files, top, backend="verible") # named only when it matters
```

### 2.2 Supported tools

| Category | Backends | Agent interface | Returns |
|---|---|---|---|
| **lint** | verilator (default) | `lint(files, top)` | `[{file, line, severity, code, message}]`, returned in full |
| **sim** | verilator (default), xsim | `sim(files, top, tb)` | summary plus `wave` / `log` handles |
| **debug** | fst (default), vcd | `first_mismatch(wave, ref, dut)`<br>`value_at(wave, sig, cycle)`<br>`value_range(wave, sig, from, to)`<br>`signals(wave, scope)` | a few lines |
| **view** | gtkwave | `show_wave(wave, signals, center_cycle)` | opens a window for a person |
| **synth** | yosys / vivado | `synth(files, top, mode="estimate"\|"full")` | summary plus a `report` handle |

**The two synth backends are not interchangeable.** They are a precision-versus-
speed trade-off, so the choice is expressed as `mode` rather than hidden behind
`backend`:

| mode | Backend | Speed | Purpose |
|---|---|---|---|
| `estimate` | Yosys | seconds | rough LUT / FF counts inside the agent's loop |
| `full` | Vivado | minutes → `job_id` | **the only source of truth**: real timing, utilization, bitstream |

**debug and view are two paths over one waveform handle.** The agent queries
programmatically (hundreds of MB → one line of context); the person uses
GTKWave. The agent never looks at a waveform picture.

`show_wave` has the daemon generate a `.gtkw` save file first (signals, scope
and zoom position preset), then computer-use runs `gtkwave dump.fst view.gtkw`.
Without a `.gtkw`, someone has to click through the signal tree by hand: slow
and error-prone.

### 2.3 Modularity: a two-level registry

```python
# ic_core/tools/lint/__init__.py  — the category contract
class LintIn(BaseModel):
    files: list[str]
    top: str
    backend: str = "verilator"

class LintOut(BaseModel):
    issues: list[Issue]

@category("lint", In=LintIn, Out=LintOut, cacheable=True, long_running=False)
class LintCategory: ...
```

```python
# ic_core/tools/lint/verilator.py  — the backend implementation
@backend("lint", "verilator", default=True)
class VerilatorLint:
    version_cmd = ["verilator", "--version"]   # recorded into meta.json

    def run(self, i: LintIn) -> LintOut:
        ...
```

The registry is the single source of truth. Everything below is generated from
it and needs no manual synchronisation:

| Generated | From |
|---|---|
| MCP tool list + JSON Schema | the category's pydantic models |
| CLI subcommands (`ic lint --files ... --top ...`) | the same (typer reflection) |
| HTTP endpoints | the same (FastAPI consumes pydantic directly) |
| `meta.json`'s `inputs` field | `In` fields + backend name + backend version |
| Sync path or job path | the category's `long_running` |

| To do this | Files touched |
|---|---|
| Swap or add a backend in the same category (Verible, xsim, VCD) | **1** (the new backend file plus its decorator) |
| Add a new category (`fmt`, `cov`, …) | **1 folder** (contract plus at least one backend) |
| Change the daemon / MCP server / CLI | **0** |

---

## 3. Daemon layer

### 3.1 Persistence: one run, one directory

Running a tool once produces one directory on disk:

```
.ic/runs/2026-09-21/143052-a3f21c-sim/
    meta.json      what ran, whether it worked, how long it took
    out.json       the summary the agent received
    artifacts/     wave.fst (184 MB), sim.log (12 MB)
    work/          Verilator obj_dir, Vivado project, etc.
```

**Why group by run rather than by `tools/lint/`, `tools/sim/`**: the question
asked afterwards is almost always "what did that last run do". One sim produces
both a waveform and a log; they are products of the same event and belong
together. Splitting by tool files two halves of one problem in two places.

**Directory name = time + run_id + category**, so `ls` sorts by time for free.
The date level makes retention a matter of `rm -rf runs/2026-08-*`.

**Three levels inside a run directory, with different lifetimes**:

| Level | Contents | Size | Retention |
|---|---|---|---|
| `meta.json` + `out.json` | what ran, result summary | KB | **forever** |
| `artifacts/` | wave.fst, log, report | MB–GB | retention policy |
| `work/` | obj_dir, Vivado project | GB | **reclaimed first** |

After `artifacts/` is deleted, `meta.json` is still there and the history
remains queryable.

**meta.json**

```json
{
  "run_id": "a3f21c",
  "category": "sim", "backend": "verilator", "backend_version": "5.020",
  "inputs": { "top": "npu_systolic_array", "tb": "tb_matmul",
              "files": [{"path": "...", "sha256": "..."}] },
  "state": "succeeded",
  "started_at": "...", "duration_s": 42.1, "exit_code": 0,
  "artifacts": [{"name": "wave.fst", "bytes": 184320000}],
  "parent_run": null
}
```

Input files are recorded by sha256, not copied. `parent_run` lets a debug run
point back at the sim that produced the waveform, so provenance chains up.

**A handle is run_id + filename**: `a3f21c/wave.fst`. It maps straight to a
path on disk, so it stays valid across a daemon restart with nothing to
rebuild.

**index.db** (SQLite) holds only the queryable columns: `run_id, ts, category,
backend, top, state, duration, dir, parent`. It answers "which run was the last
sim of `npu_systolic_array`" without walking the whole tree. **The truth is in
meta.json; index.db can be deleted and rebuilt at any time.**

**Write atomicity**: write to `143052-a3f21c-sim.tmp/` first, then rename. An
interrupted run keeps its `.tmp` suffix and GC removes it, so a half-finished
result can never be mistaken for a valid one.

> **No caching at this stage.** The same inputs run again is simply another run
> and produces a new run directory. Should caching be wanted later, the
> `inputs` in `meta.json` already carry sha256 values, so a content-hash index
> can be built from them without changing the storage structure.

### 3.2 Job lifecycle

**Recommendation: write a thin layer, roughly 150 lines. Do not introduce a job
framework.**

```
queued → running → succeeded
                 → failed
                 → cancelled
```

| Component | Approach |
|---|---|
| Execution | `asyncio.create_subprocess_exec`, with stdout/stderr redirected straight to files (never into memory) |
| State | a single SQLite table: `job_id, tool, state, pid, started_at, ended_at, artifact_dir` |
| Cancellation | record the pid, `terminate()`, then `kill()` on timeout |
| Restart recovery | scan the table at startup; anything `running` whose pid is gone becomes `failed` |
| Concurrency limit | `asyncio.Semaphore` (a single Vivado instance is several GB; two on a desktop means swapping) |

**The agent side uses long polling, not status polling**:

```
synth(design_id)                   → job_id
wait_for_job(job_id, timeout=300)  → the result if finished, otherwise still_running
```

A 40-minute job produces about 8 calls, each of which carried information.
Plain polling (asking every 30 seconds) piles 80 `{"status":"running"}` objects
into the context. Blocking outright risks a client timeout, and a dropped
session leaves the result unclaimable.

> **Why not Celery / RQ**: both need a broker (Redis / RabbitMQ). For a single
> machine, a single user and jobs numbered in the single digits, a broker is
> pure overhead.
> **Why not Huey**: it can use SQLite and is lighter than Celery, but it still
> brings a worker model and a task registration mechanism, when all that is
> wanted is "spawn a subprocess and remember it".

---

## 4. Notes

### 4.1 Limits

| Item | Limit |
|---|---|
| **Verilator** | Two-state (does not model X propagation); cycle-based (no `#10` delays); testbenches need C++ or `--binary` to run SV. **Existing testbenches may need changing — this is the plan's only technical risk.** |
| **Yosys** | Estimates only. Producing a Zynq-7000 bitstream still requires Vivado for place and route. |
| **Vivado state** | An open project cannot be serialised; the daemon has to hold the process. |
| **GTKWave** | GUI only. An agent cannot get information out of it, only open it for a person. |
| **Waveform format** | FST (`--trace-fst`). VCD files are 10–100x larger and have no random access. If no usable FST parser exists for Python, fall back to VCD. |
| **Memory** | A single Vivado instance is several GB, so a concurrency limit is mandatory. |

### 4.2 Out of scope (revisit if this becomes an ASIC or multi-person project)

OpenROAD (P&R), OpenSTA / PrimeTime (signoff STA), DRC / LVS, FlexLM license
pools, LSF / SGE / k8s scheduling, OAuth + RBAC + auditing, MCP Apps
interactive viewers.

### 4.3 Development order

Each stage is independently usable; none waits for the rest.

| # | Content | What the agent gains | Needs the daemon? |
|---|---|---|---|
| 1 | Lint | knows within seconds whether an RTL edit is sound | no |
| 2 | Sim + run store | can run regressions and see which test broke | **yes** (run directories and handles) |
| 3 | Debug (FST queries) | locates the failing cycle and signal on its own | **yes** |
| 4 | GTKWave + `.gtkw` | can open the problem in a window for a person | no |
| 5 | Synthesis (Yosys + Vivado job) | area and timing feedback | **yes** |
| 6 | Skills + CLI packaging | one skill behaves the same in Claude and in pi | — |

P1 takes a day and pays off immediately. P3 is both the hardest and the most
valuable. The daemon is first needed at P3.

### 4.4 Open questions

| # | Question | Status |
|---|---|---|
| Q1 | How much SystemVerilog do the existing testbenches use, and can Verilator take it? | **Resolved — it can.** `--binary --timing` runs the existing SV testbenches such as `tb_npu_pe` directly, with no change to any testbench. The "only technical risk" noted above did not materialise. |
| Q2 | How should Python parse FST? Fall back to VCD if no usable package exists. | **Resolved — no fallback needed.** There is indeed no usable pure-Python FST package, but GTKWave ships `fst2vcd`, and the `view` category already depends on GTKWave. The `fst` backend converts once and caches the result under `.ic/cache/`; every subsequent query is a plain streaming VCD scan. Simulations still write FST, so what is stored stays small. |
| Q3 | What does MCP 2026-07-28 actually require for timeout / progress? | Not investigated. The implementation sidesteps it with long polling: `IC_MCP_WAIT_S` defaults to 90 seconds and returns a `job_id` on timeout, so it does not depend on client timeout behaviour. |
| Q4 | What is the target client's default tool timeout? It sets the long-poll window. | Not investigated, and no longer blocking: a timeout degrades into more calls rather than a failure. |

---

## 5. Implementation results

Implemented under `tools/ic/`. Usage and extension are documented in
[`docs/ic-design-tools/README.md`](../ic-design-tools/README.md); per-tool
reproduction steps are in
[`docs/acceptance/reproduce.md`](../acceptance/reproduce.md).

### Where the implementation follows this plan

All eight ops work: `lint`, `sim`, `signals`, `first_mismatch`, `value_at`,
`value_range`, `show_wave`, `synth`. The layer boundaries, the run directory
structure, the handle form (`run_id/filename`), `meta.json` as the truth with a
rebuildable `index.db`, `.tmp` atomic writes, no caching at this stage, and a
hand-written job layer of roughly 150 lines instead of a job framework — all as
described above.

### Where it differs

| Item | Plan | Implementation | Why |
|---|---|---|---|
| Schema granularity for debug | one In/Out pair per category | a category may hold several **ops**, each with its own In/Out | Folding debug's four queries into one schema would force the agent to dispatch on an `op` field, and no single tool description could be precise about all four. `lint`/`sim`/`synth`/`view` remain single-op, so the contract still belongs to the category. |
| `.gtkw` contents | "signals, scope and zoom position preset" | also needs `@22`/`@28` trace-flag lines and a vector's `[7:0]` bit range | Without them GTKWave **silently ignores** the signal names and opens an empty window — which looks like success. |
| Where the waveform comes from | unspecified | a probe module is generated per run and `bind`-ed into the testbench | None of the existing testbenches call `$dumpfile`/`$dumpvars`. Editing eight of them would mean the tools modified the very thing they exist to observe. |
| Scope of run records | every invocation produces a directory | read-only queries (`records_run=False`) do not | One debugging session issues dozens of waveform queries; a directory each would bury the sim runs that matter. |
| How pi is integrated | "Skills + CLI packaging" | an extension (native tools) plus the skill plus the CLI | Same direction. pi does have no built-in MCP, but it does have `registerTool`, so the tools can be native rather than merely prompted through the CLI. |

### Not yet verified

`synth --mode full` (no Vivado on this machine), the board itself, and measured
performance on gigabyte-scale waveforms. See the final section of reproduce.md.
