# synth_csd_multiplier

Predeclared objective: area. Two substantive configurations and both
architectures are fixed in config.json before any physical measurements.
All variants have identical registered boundaries, II=1 and latency=1.

```powershell
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 17 --container codex-sandbox-agent-workspace --verify-only
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 17 --container codex-sandbox-agent-workspace
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

Use `network_study.py --tool 17 --configuration config-bitwise.json --container
codex-sandbox-agent-workspace --verify-only` for the bounded 300-second bitwise
retry. It keeps the original widths, constants, architectures and area objective.
Study fe89db proves all 24 bits of the 16-bit case; the 32-bit case remains
inconclusive. Old whole-output timeouts remain part of the evidence history.

Physical study 7830ae completes three width16 pairs: LUT reduction 76%,
throughput gain 102.60%, six timing audits pass. Width32 remains unknown in both
bitwise and macc retries (eb1594); a single passing size does not qualify a family.
config-macc.json retains the original width32/constant65535 area comparison.

`config-prefix.json` retries the original unresolved case(s) with ordered
output-bit SAT. It uses already-proved lower-bit equalities and still requires
every bit, complete source guards and a successful bounded process. Use
`--configuration config-prefix.json` with this module's study command. Full
verification now passes; physical gain still requires all three measured pairs.
Historical failures and the original objective remain in the evidence.
