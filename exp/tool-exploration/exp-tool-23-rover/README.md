# synth_popcount: registered architecture study

This is one distinct generator, with linear and hierarchical implementations.
Configurations and the throughput objective are predeclared in config.json.
All cases use registered boundaries, II=1 and latency=1; all FPGA resource
classes must be retained. Three paired physical repeats are required.

Run from the repository root:

```powershell
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 23 --container codex-sandbox-agent-workspace --verify-only
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 23 --container codex-sandbox-agent-workspace
```

Independent Python-oracle tests cover both substantial configurations, zero,
all-one, walking-bit and 1,024 seeded random inputs. SAT equivalence and 8,192
random vectors plus directed cases gate physical measurements. Unsupported or
timed-out proof attempts remain failures, and do not disappear from acceptance.

Paper connection: rover, as documented in survey.md. This is a bounded
architecture adaptation, not a reproduction of the paper's learned search.
Both physical comparisons are complete. Width64 grows LUTs 6.25% and loses
3.66% throughput; width128 grows LUTs 2.18% and loses 9.21% throughput. All 12
timing audits pass, and the family does not qualify.

Use network_study.py --tool 23 --configuration config-macc.json with the same
container option for the original width128 retry after word SAT timed out.
The retained retry closes the width128 proof and measurement gap without
removing the earlier timeout or either regression from acceptance.

Run `controls.py` for the shared operation 23-28 mutation suite. It checks both
declared sizes for population count, priority, leading-zero, shift, one-hot and
argmax networks. Correct trees pass; architecture-specific operator and tie-rule
mutations fail. These are vector failure controls, not extra physical evidence.
