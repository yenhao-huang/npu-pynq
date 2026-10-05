# A/B test: what the tools change

One question, asked so the answer is a number: **when an agent is given
`tools/ic/`, what changes?**

Three dimensions, one table. Everything below exists to fill it in.

|             | Success rate | Cost per success | ADP ratio |
| ----------- | ------------ | ---------------- | --------- |
| No tools    |              |                  |           |
| All tools   |              |                  |           |

- [The two arms](#the-two-arms)
- [Metrics](#metrics)
- [Task set](#task-set)
- [Rules the solution must obey](#rules-the-solution-must-obey)
- [Grading procedure](#grading-procedure)
- [What is recorded](#what-is-recorded)
- [Known limitations](#known-limitations)

## The two arms

Same model, same prompt, same task. The only difference is what it can call.

| Arm | Tools available |
| --- | --- |
| **No tools** | None. The model writes RTL from the description and cannot execute anything — no simulator, no linter, no shell. |
| **All tools** | The full `tools/ic/` surface: `lint`, `sim`, `signals`, `first_mismatch`, `value_at`, `value_range`, `synth`, `ppa`. |

`show_wave` is excluded from the "all tools" arm. It opens a GUI for a person
and nothing can be read back from it, so in an unattended run it can only waste
a call.

`ppa` **is** in the arm. The question being asked is what the toolchain as a
whole changes, and singling one tool out would answer a question nobody asked.
The consequence is that the ADP ratio measures a treatment effect — an agent
that can see its own PPA may spend budget improving it — so ADP and cost per
success are read together, never on separate pages.

## Metrics

### Capability — success rate

- **pass@1** (primary). One attempt, and `make -C src/test lint sim` passes
  against the grading testbench. This is the headline number.
- **pass@k**, k = 5. The same problem attempted five times independently. The
  gap between pass@1 and pass@5 separates *capability* from *consistency*: a
  model that succeeds one time in five is not equivalent to one that succeeds
  every time.

Five independent runs per problem per arm. Report the median and the
interquartile range, never a single run — the variance between runs of the same
model is large enough to invert a conclusion.

### Efficiency — cost per success

Both primary metrics divide by successes rather than by attempts, so that
failing cheaply cannot look like an advantage:

- **Token cost per success** = total tokens across every attempt in the arm
  ÷ successes.
- **Wall-clock per success** = total elapsed time across every attempt ÷
  successes.

The two are reported side by side and never summed. They can move in opposite
directions, and that trade-off is a result: the tool arm may spend fewer tokens
while taking longer, because a simulation costs seconds of wall clock and
almost no context.

Supporting:

- **Tokens split into input / output / cache read.** Totals alone mislead. The
  tool arm's input is larger but mostly cache-hit, so what is actually paid for
  differs from what is counted.
- **Peak context occupancy.** The claim that nothing large is ever returned is
  only testable here: an arm that reads a 12 MB `sim.log` directly shows up in
  this number and nowhere else.
- **Iteration count** — edit→verify loops until the agent stops.

### Quality — ADP

**ADP (area-delay product)**, from `ic ppa` on each passing solution:

    ADP = area.cell_area_um2 x (1000 / timing.fmax_mhz)     [um^2 x ns]

Area and speed trade against each other, so neither alone can rank two designs:
a solution that grows to run faster looks better on one and worse on the other.
Their product collapses that trade-off into one comparable quantity.

Reported as a **ratio between the arms**, per problem:

    ADP ratio = ADP(all tools) / ADP(no tools)

No human reference implementation is needed, and none is used. The other arm is
the baseline, which is what an A/B test compares anyway. Where a problem has
successes in only one arm it is excluded from the ratio and counted in the
success rate alone.

Every `ppa` call in the grading pass uses the **same liberty library, the same
`--clock-period-ns`, and the same `--input-activity`**. Power is estimated from
assumed switching activity rather than from a trace, and area and fmax depend on
the library, so numbers taken under different settings are not comparable.
`mode="estimate"` throughout, for cost; `placed` only if a result is contested.

Supporting: `area.cell_area_um2` and `timing.fmax_mhz` reported separately
alongside the product, so a surprising ADP can be attributed to one or the
other. `power.total_w` is recorded but is not a metric — it is pinned to the
activity assumption and would only dilute the product.

## Task set

**RTLLM v2.0** — 50 hand-crafted designs in four classes: Arithmetic, Memory,
Control, Miscellaneous. Each ships a natural-language description, a testbench,
and a verified reference implementation.

Pin the commit. The upstream repository has moved past v2.0 (a v2.1 with
corrected descriptions and testbenches was published in August 2026), and a task
set that shifts mid-experiment invalidates the comparison.

What the agent is given: **`design_description.txt` only.** Not `testbench.v`,
not `verified_verilog.v`.

The tool arm can still run `ic sim`, but against the *held-out* stimulus
described below — never against the file the grader uses.

## Rules the solution must obey

The metrics above assume the tests are trustworthy. These two rules are what
makes that assumption hold. Both are enforced mechanically; neither is scored as
its own metric, because a violation does not make a solution lower quality — it
makes it not a solution.

1. **No answers derived from the testbench.** Hard-coding expected outputs,
   building a lookup table keyed on the test stimulus, or special-casing the
   values the testbench happens to drive. Such a solution passes, synthesises
   tiny, and scores an excellent ADP while implementing nothing.
2. **No deleting what the tests do not cover.** ADP rewards small and fast, so
   removing untested logic — pipeline registers, saturation and overflow
   handling, reset behaviour, unconnected ports — improves the score while
   breaking the design.

Enforcement is by construction rather than by inspection, so it does not depend
on anyone enumerating the ways to cheat:

- The agent never sees `testbench.v` or `verified_verilog.v`.
- Grading runs on **held-out stimulus**, generated per problem by simulating
  `verified_verilog.v` against fresh random vectors plus deliberate boundary
  cases: saturation and overflow limits, reset timing, zero and maximum
  operands, back-to-back transactions. Rule 1 fails here because the lookup does
  not cover the unseen vectors; rule 2 fails because the boundary cases exercise
  exactly what was removed.
- Test files are restored from the pinned checkout before grading, so anything
  the agent changed under the test directories has no effect.

## Grading procedure

Per attempt, run by the harness — never by the agent, and never from the agent's
own report of what it achieved:

```bash
# 1. Restore everything the agent must not have influenced.
git checkout $PINNED -- <testbench and stimulus paths>

# 2. Functional verdict: shipped testbench plus the held-out vectors.
make -C src/test lint sim

# 3. Quality, for passing solutions only.
ic ppa --files <solution> --top <module> \
       --clock-port clk --clock-period-ns $PERIOD --compact
```

A solution that fails step 2 is a failure regardless of what the agent
reported. ADP is computed only over passing solutions: the area and speed of a
circuit that does not work are meaningless.

## What is recorded

One row per attempt, so that any number in the headline table can be traced
back:

| Field | Why |
| --- | --- |
| problem id, arm, repeat index | keys |
| passed (bool) | feeds pass@1 / pass@k |
| tokens: input, output, cache read | the cost split |
| wall-clock seconds | the other cost |
| iteration count | the feedback-loop claim |
| peak context bytes | the "nothing large is returned" claim |
| tool calls, by tool name | which tool did the work |
| `ppa` area, fmax, ADP | quality |
| failure class | see below |
| diff touched test paths (bool) | free signal, not a metric |

Failure classes, for the attempts that did not pass: wrong approach, looped
without progress, gave up, exceeded budget, **claimed done**. The last one costs
nothing to record and is the only trace left of an agent that reported success
it did not have.

`tool calls, by tool name` matters more than it looks. If the tool arm turns out
never to have called `ppa`, then an ADP difference came from somewhere else, and
this column is the only way to find that out afterwards.

## Known limitations

Recorded before the run, so that the results are read with them in view.

- **RTLLM specifications are prescriptive.** Several dictate internal
  implementation down to register names, widths and iteration counts, so correct
  solutions resemble each other closely. ADP variance will therefore be narrower
  than the metric can express, and much of what remains reflects poor
  implementation rather than genuine design choice. Problems with real design
  freedom — the arithmetic and miscellaneous classes — carry the ADP result;
  the control class largely will not.
- **Ceiling effect.** The designs are module-scale. If both arms pass most
  problems on the first attempt, the capability dimension has no room to show a
  difference, and the finding is about the task set, not about the tools.
- **Small designs understate the tools.** `first_mismatch`, `value_at` and
  `value_range` earn their place on stateful, multi-cycle designs where the
  divergence point is not obvious. A ten-line module does not need a waveform
  query, so this task set measures the verification loop more than the debugging
  tools.
- **Contamination.** RTLLM has been public since 2023, reference
  implementations included, so the models under test have likely seen the
  solutions. This inflates both arms roughly equally — the *difference* survives
  it, the absolute success rates do not, and they should not be compared against
  published numbers.
- **No debugging tasks.** Every problem is written from scratch. "Here is a
  broken design, fix it" is absent, which is the scenario the waveform tools
  exist for.
- **ADP is pre-route and technology-neutral.** `ic ppa` reports standard-cell
  numbers for the library it was given. It ranks two implementations; it says
  nothing about the Zynq-7020.

The task set, the grading script and these metric definitions are **frozen
before the first scored run**. Prompts may be tuned on problems held aside for
development, which do not enter the results. If anything here changes after
results are seen, the change and its reason are recorded rather than quietly
applied.
