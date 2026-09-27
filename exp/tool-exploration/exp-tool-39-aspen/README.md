# netlist_profile

Run the shared substantive study:

```powershell
.venv/Scripts/python.exe exp/tool-exploration/netlist_study.py --container codex-sandbox-agent-workspace
```

The study generates real Yosys generic/xc7 netlists for shift and circular FIFOs
at 16x64 and 32x128, and arithmetic graphs for independent/shared MCM at 16/32
bits. JSON digests bind each consumer to its producer. Generic operator counts,
mapped primitive classes and routed Vivado resource counts are different scopes.

critical_cone reports endpoint fan-in and unweighted logic depth, stopping at
state/memory boundaries; it does not identify a routed critical path by itself.
fanout_analysis counts sink pins, including clock/reset/enable pins, rather than
capacitance. memory_inference separates logical bits, RAM/SRL primitives and
FF bits; control FFs are not automatically classified as FIFO data storage.

The aspen connection is diagnostic feedback around the documented optimization
workflow, not a claim to reproduce a paper's entire search algorithm. The
circular-storage study directly follows the scalar-replacement storage choice.
Malformed JSON, digest changes and combinational cycles have negative tests.
