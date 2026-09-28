# synth_leading_zero: registered architecture study

This is one distinct generator, with linear and hierarchical implementations.
Configurations and the throughput objective are predeclared in config.json.
All cases use registered boundaries, II=1 and latency=1; all FPGA resource
classes must be retained. Three paired physical repeats are required.

Run from the repository root:

```powershell
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 25 --container codex-sandbox-agent-workspace --verify-only
.venv/Scripts/python.exe exp/tool-exploration/network_study.py --tool 25 --container codex-sandbox-agent-workspace
```

Independent Python-oracle tests cover both substantial configurations, zero,
all-one, walking-bit and 1,024 seeded random inputs. SAT equivalence and 8,192
random vectors plus directed cases gate physical measurements. Unsupported or
timed-out proof attempts remain failures, and do not disappear from acceptance.

Paper connection: prefixrl, as documented in survey.md. This is a bounded
architecture adaptation, not a reproduction of the paper's learned search.
All declared physical comparisons are complete. Tree64 saves 20.55% LUTs but
loses 3.39% throughput; tree128 saves 6.49% LUTs and loses 4.00% throughput.
Binary-search64 is unchanged in LUTs and loses 8.29% throughput;
binary-search128 grows LUTs 11.69% and loses 16.11% throughput. The family does
not qualify at both sizes.

## Bounded same-case retry

Run `python exp/tool-exploration/network_study.py --tool 25 --configuration
config-retry.json --container codex-sandbox-agent-workspace` as one command.
The three unfinished original cases keep their throughput objectives and Basic
flow; up to three attempts per physical measurement preserve every failure.
Study 684174 completes three pairs per case and 18 passing timing audits.
Tree64 reduces LUTs 20.55% with throughput -3.39%; binary-search64 is unchanged
in LUTs with throughput -8.29%; binary-search128 grows LUTs 11.69% and loses
16.11% throughput. The tree's area win does not change its declared objective,
and this family does not qualify at both widths.
