# Reduction architectures: substantive physical study

Operation: `synth_adder_tree`. ROVER's mixed-width arithmetic exploration
motivates comparing structurally different implementations. This generator
implements unsigned modular sums using serial carry-propagate, balanced
carry-propagate and carry-save reduction. These are variants of one tool,
not three tools counted toward the 50-tool target.

Predeclared configurations: 16 operands of 16 bits, then 16 operands of 32 bits.
Objective: throughput at identical one-cycle latency and II=1; report every
resource class and all three variants. Baseline is natural serial arithmetic,
without keep/dont_touch attributes. All variants use the same input/output
register wrapper. Three implementation repeats per variant and configuration
are reproducibility checks, not statistically independent samples.

```powershell
.venv/Scripts/python.exe exp/tool-exploration/exp-tool-14-rover/study.py
```

The runner checkpoints successful calls under ignored `output/<source-digest>`.
Failed calls remain visible and are retried on resumption. Do not launch a second
copy while an existing study is running. The current runner does not yet validate
tool-version changes on resume; use a fresh output namespace after environment
changes. Full expanded acceptance also requires formal coverage, other
microarchitecture families and the remaining tools.

Hypothesis: carry-save compression may shorten long carry-propagate reductions.
Balanced syntax alone may synthesize to nearly the same netlist. Both outcomes
must remain in the report; a small timing variation is not a qualifying win.
