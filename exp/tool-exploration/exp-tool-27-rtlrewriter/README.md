# synth_onehot_mux: registered architecture study

This is one distinct generator, with linear and hierarchical implementations.
Configurations and the throughput objective are predeclared in config.json.
All cases use registered boundaries, II=1 and latency=1; all FPGA resource
classes must be retained. Three paired physical repeats are required.

Run from the repository root:

```powershell
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 27 --container codex-sandbox-agent-workspace --verify-only
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 27 --container codex-sandbox-agent-workspace
```

Independent Python-oracle tests cover both substantial configurations, zero,
all-one, walking-bit and 1,024 seeded random inputs. SAT equivalence and 8,192
random vectors plus directed cases gate physical measurements. Unsupported or
timed-out proof attempts remain failures, and do not disappear from acceptance.

Paper connection: rtlrewriter, as documented in survey.md. This is a bounded
architecture adaptation, not a reproduction of the paper's learned search.
Results are pending; a structural change alone is not a measured PPA gain.
