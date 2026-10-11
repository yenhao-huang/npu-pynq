# Modular A/B harness

Run commands from `exp/ab_test/`. Python 3.10 or newer is required.

```text
src/
  main.py                   CLI, problem selection, scheduling
  config.py                 Explicit runtime settings
  runner.py                 Connect agent, grader, and recorder
  problems/
    catalog.py              Problem discovery and reference preparation
    patches.py              Known dataset corrections
  agent/
    rounds.py               Single-round and multi-round execution
    context.py              Conversation ID, history, session continuation
    prompt.py               Initial and continuation prompts
    llm.py                  Claude CLI adapter
    codex.py                Codex CLI adapter (gpt-* models)
    parse.py                Event parsing and usage aggregation
  grade/
    evaluator.py            Correctness gates and PPA evaluation
    shipped.py              Dataset testbench evaluation
    heldout.py              Deterministic differential checks
    ports.py                Yosys port extraction
    ppa.py                  Fixed-constraint area, timing, power
    pareto.py               Nondominated candidates and grouped frontiers
  logging/
    recorder.py             Context, raw events, snapshots, results
../problems/                Dataset files, separate from Python modules
```

`problems` has no dependency on agent execution. `agent` does not import the
grader. `runner` receives a model with `generate()` and a grader with
`evaluate()`, so either can be replaced in tests or by another backend.
Importing modules does not discover problems or execute external commands.

## Usage

```bash
# Inspect the dataset without invoking a model.
python -m src.main --list

# Single round, one repetition per arm.
python -m src.main adder_8bit --rounds 1 --reps 1 --arms notools tools

# Three rounds in one conversation per attempt.
python -m src.main adder_8bit --rounds 3 --reps 1 --arms notools tools

# Configure dataset, tool repository, and model independently.
python -m src.main adder_8bit --problems-dir ./problems --repo "$AB_TOOL_REPO" --model sonnet --rounds 3

# gpt-* models run through the Codex CLI (--backend auto picks it; claude|codex forces one).
python -m src.main adder_8bit --model gpt-6-sol --rounds 1 --reps 1 --arms notools tools

python -m unittest discover -s tests -v
```

`CodexModel` runs `codex exec --json` with shell execution disabled, so files change only
through `apply_patch`; the `tools` arms attach the same `ic` MCP server (without `show_wave`,
tools auto-approved). Its raw JSONL is kept and Claude-style `assistant`/`result` events are
appended, so parsing, round control and grading are shared. Multi-round resumes the Codex
thread recorded in round 1. Codex has no per-call budget flag, so `--budget` does not apply
and `cost_usd` is 0; token usage is still recorded.

The previous script path, `python src/ab_test_single_round.py`, remains a thin
compatibility entry point. Use `--rounds N` for multiple rounds. CLI flags take
precedence over `AB_ROUNDS`, `AB_MODEL`, `AB_BUDGET`, `AB_REPS`, `AB_PAR`, and
`AB_ARMS`. Defaults are one round, three independent repetitions, medium effort, and six parallel attempts.
Use `--repetitions` (alias `--reps`) for independent attempts and `--rounds` for
conversation rounds within each attempt. `--effort` (or `AB_EFFORT`) is passed
explicitly to each Claude invocation and saved in `config.json`.
Without problem IDs, all discovered problems run. `--budget` is the budget
passed to each Claude invocation; it is not a total experiment spending cap.

The Claude CLI must be authenticated. Source `.ab_env` after running
`setup/setup.sh` to select the pinned IC tools checkout. Grading requires `iverilog`, `vvp`,
`yosys`, and an IC tool environment with the configured Nangate45 files.
`--repo` defaults to the `AB_TOOL_REPO` environment variable when set.

## Round and context semantics

One round is one Claude CLI invocation, which may contain multiple model and
tool turns. In multi-round mode, the first round creates a session and later
rounds resume that session in the same temporary workspace. The continuation
asks for another correct implementation exploring a different power, delay, or
area tradeoff. Each round submits one candidate in `solution.v`; intermediate
alternatives are not independently graded.
Single-round mode disables session persistence. Multi-round mode uses the
CLI's persisted sessions; their retention is controlled by the CLI.

`notools` exposes Write in single-round mode, and Write/Read in multi-round
mode so the model can inspect the previous design, matching the existing
multi-round harness. `tools` and `tools_directed` expose Write/Read/Edit and
IC MCP tools, excluding `show_wave`. `tools_directed` adds the prescribed
workflow to the first prompt. Reads of the dataset and original private
reference directory are disallowed in the Claude tool configuration.

After all model rounds finish, every snapshot is graded independently. Hidden
testbench output and references are never added to a continuation prompt.
The result contains a Pareto frontier of the observed candidates. Hidden grades
are used only after generation, never to guide the model's continuation.
Known dataset corrections and the explicit `radix2_div` held-out skip remain;
unexpected port extraction failures now fail correctness evaluation.

Timeouts, nonzero model exit codes, API errors, and missing result events stop
further rounds and are recorded as failures. An older solution left in the
workspace cannot make an unsuccessful model round pass. Grading errors are
recorded per round, allowing other snapshots to be evaluated.

## Logs

Each invocation creates a unique directory below `results_modular/` (or
`--output`), preventing results from different settings from being reused.
Each attempt contains:

- `config.json`: effective settings.
- `context.json`: session identity and submitted prompts / final assistant text.
- `roundN.jsonl`: raw CLI events, including emitted tool calls and usage.
- `roundN.stderr.txt` and `roundN.response.json`: process diagnostics.
- `roundN.v`: solution snapshot, when present.
- `roundN.grade.json`: correctness and PPA results.
- `frontier.json`: nondominated candidates for this attempt.
- `result.json`: aggregated usage, per-round results, `final_round`, and `pareto_frontier`.

The run directory also contains `frontier.json`, grouping candidates by problem.
Each problem has a combined frontier and separate `by_arm` frontiers across
repetitions. `solution_path` is relative to the run directory and points to the
preserved round snapshot. Dominated snapshots remain available for auditing.

## Pareto objectives

Minimize `area_um2`, `power_w`, and `delay_ns = 1000 / fmax_mhz`. Keep
`fmax_mhz` and `wns_ns` as diagnostic measurements; WNS does not affect
membership and a negative WNS is not an additional feasibility filter.
Pareto membership uses only those three objectives, without a scalar score or weighting.
For report presentation, `top3_pdap` sorts the frontier by
`PDAP = power_w * delay_ns * area_um2` ascending and retains up to three entries.
Full frontiers are preserved; PDAP is not used in the agent prompts or dominance test.
Ties use stable input order. PDAP values are compared only within the same problem.
Run summaries also include `top3_pdap_by_arm`.

Only functionally passing candidates with `ppa_ok=true` and finite, strictly
positive values for all three objectives are eligible. Held-out evaluation
must pass unless the problem has an explicit documented skip. Missing metrics,
failed PPA processes, and unexpected held-out errors cannot enter the frontier.

A candidate is removed only when another eligible candidate has values no
larger for all three objectives and strictly smaller for at least one. Exact
ties are retained with their provenance. Comparisons use unrounded values,
without tolerance. Candidates from different problems are never compared.
This is the frontier of sampled round snapshots, not a claim of global
optimality. The attempt's top-level `passed` means at least one eligible
candidate is retained; `final_round` separately records the last round's
outcome, including a possible later failure.

`context.json` records the harness-visible conversation. Internal system
prompts, CLI compaction, and unexposed model context are not reconstructed;
raw event streams retain the details emitted by the CLI. Logs are written
after each response, and context is also saved before invoking the model.
Temporary workspaces are removed after grading. There is no automatic retry
or resume of interrupted harness runs; re-running creates a fresh experiment.
