# timing_constraint_audit

`audit_historical_records.py` separately authenticates older pre-coverage
records against their preserved local run envelopes, exact source files, Tcl,
parsed metrics and raw report digests. It does not infer missing clock/register
coverage or `check_timing` results. Run:

```powershell
.venv/Scripts/python.exe exp/tool-exploration/exp-tool-43-rtlrewriter/audit_historical_records.py
```

Paper connection: RTLRewriter, ICCAD 2024,
https://arxiv.org/html/2409.11414v1. Controlled physical evaluation motivates
auditing measurement assumptions. The tool is a local verification extension,
not a constraint-inference algorithm reproduced from the paper.
Vivado's installed `help check_timing` and `help all_registers` define the
actual timing coverage queries. A real probe report established parser syntax.

Run `python exp/tool-exploration/exp-tool-43-rtlrewriter/run.py` on the licensed
host. The experiment routes 16/32-bit registered adders, audits their actual
reports, then routes a design with a second unconstrained clock input. That
negative design must not receive a clean coverage audit. All outcomes are saved.
These runs validate the audit tool, not a new PPA improvement family.

The audit binds exact RTL hashes, top, tool/build, reported slack/period,
routed setup report and coverage reports from one run. It rejects missing
checks, mixed reports, contradictory summaries, unclocked cells, clock ambiguity,
loops and unsupported latches. Report digests are returned. A negative setup
slack is explicitly not timing closure even when constraint coverage passes.

Scope is single-clock register-to-register setup in an OOC design. Top-level
I/O delays, board constraints, hold closure and protocol correctness remain
excluded. Historical records without new coverage reports are unaudited, not
silently upgraded. Parser fixtures in tests are explicitly synthetic.

## DSP48E1 inactive storage

The coverage report now lists every cell outside ic_clock and its actual DSP
properties. Only DSP48E1 with USE_DPORT=FALSE/0, USE_MULT=NONE and all other
register attributes zero may exclude the unused ADREG/DREG defaults. Counts,
unique identities and complete properties must agree. This is a narrow modeled
case, not an exemption for all DSPs. Active, missing and unknown properties fail.
The interpretation follows [AMD UG479, Table 2-3 and pre-adder discussion](https://docs.amd.com/api/khub/documents/gu4oRPFEh_Pm2uaAlfY6Kg/content).

Run `python exp/tool-exploration/exp-tool-43-rtlrewriter/dsp.py`. Actual control
548b77/a349d5 has 145 total/144 clocked cells and one fully bypassed DSP; it
passes. Control 70daa5/3ebf43 also has 145/144 but activates PREG on an unclocked
domain; it fails no_clock, unconstrained endpoints and the active-cell count.
No PPA gain or protocol acceptance is derived from these diagnostic fixtures.
Original reports without per-cell details retain their previous strict result.

The DSP control suite now tests unused CREG through two static OPMODE settings
and active unclocked CREG. Run `dsp.py --case unused_c` or the full `dsp.py` suite.
Direct constant-driver evidence and complete per-cell properties are mandatory;
unknown controls and any reported unconstrained endpoints remain rejected.
See `docs/goals/0928-tool-exploration/evidence/dsp-creg-controls.json`.
