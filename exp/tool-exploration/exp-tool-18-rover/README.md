# synth_mcm

Predeclared objective: area. Two substantive configurations and both
architectures are fixed in config.json before any physical measurements.
All variants have identical registered boundaries, II=1 and latency=1.

```powershell
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 18 --container codex-sandbox-agent-workspace --verify-only
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 18 --container codex-sandbox-agent-workspace
```

The independent integer-oracle tests cover maximum values, carry boundaries,
zero and seeded random words. They also validate coefficient packing. The
pipeline requires SAT plus directed/random RTL equivalence before measurement.

This is a bounded rover-inspired arithmetic graph experiment; it does not
reproduce the paper's learned search or e-graph solver. Prefix topology is
explicit; CSD is canonical signed-digit expansion; MCM uses memoized factors
of the form 2^k +/- 1. No globally optimal adder graph is claimed. Native FPGA
carry chains or multiplier inference may outperform these alternatives; all
regressions and inconclusive proofs must remain visible.
