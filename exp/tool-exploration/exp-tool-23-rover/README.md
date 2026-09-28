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
Results are pending; a structural change alone is not a measured PPA gain.

Use network_study.py --tool 23 --configuration config-macc.json with the same
container option for the original width128 retry after word SAT timed out.
Width64 is already measured and remains a regression: LUT growth 6.25%,
throughput loss 3.66%, all six timing audits pass. No declaration is removed.

Run `controls.py` for the shared operation 23-28 mutation suite. It checks both
declared sizes for population count, priority, leading-zero, shift, one-hot and
argmax networks. Correct trees pass; architecture-specific operator and tie-rule
mutations fail. These are vector failure controls, not extra physical evidence.
