# timing_constraint_audit

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
