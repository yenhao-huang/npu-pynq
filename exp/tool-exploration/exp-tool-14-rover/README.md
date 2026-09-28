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

## Proof-gated complete retry

Run `python exp/tool-exploration/exp-tool-14-rover/sweep.py --container
codex-sandbox-agent-workspace` as one command. The config-macc.json retry includes
all four original width16/32, lanes16, serial-to-balanced/compressor comparisons.
It keeps the throughput objective, 5 ns Default flow and three pairs, but adds
mandatory vectors/SAT before measurement and complete architecture_sweep envelopes.
The older study.py records remain diagnostic history; none is promoted by this
retry. Add --verify-only to suppress physical execution.

Retry 2eaf36 completes all four original comparisons and 24 timing audits.
The balanced tree yields only 2.38%/3.13% throughput gain; compressor throughput
gain exceeds 43% but LUT growth exceeds 160%. Neither qualifies the family gate.
All three paired results and the transient implementation retry are retained in
`evidence/adder-macc-physical-study.json` under the goal report.

## Arithmetic mutation controls

Run `python exp/tool-exploration/exp-tool-14-rover/controls.py`. At both original
16/32-bit,16-lane sizes, the correct balanced/compressor cores pass directed and
seeded vector checks. Replacing the first addition with subtraction or shifting
the first compressor carry by two instead of one yields concrete mismatches.
All eight expected verdicts match. These controls check failure detection; they
are not additional physical measurements or formal proofs. See the goal evidence
file reduction-mutation-controls.json.
