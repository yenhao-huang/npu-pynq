# SAT equivalence

Operation: `yosys_equivalence`. ROVER motivates checking width-sensitive
rewrites before considering their cost. This tool uses Yosys SAT, not ROVER's
e-graph implementation or certificate format.

Inputs are two self-contained combinational designs with the same top and
complete packed `x`/`y` interface. The tool flattens each separately, builds a
miter, rejects remaining state cells, and proves equality for defined binary
inputs. Width warnings, unsupported cells, timeouts and proof failures cannot
produce a successful verdict. An existing Docker container may provide Yosys;
staging uses a unique run directory and never edits the container configuration.

Local validation used Yosys 0.23 (7ce5011c24b), with a 16-bit four-operand
balanced reduction, an incorrect candidate, and a truncated input interface.
Only the correct candidate passed. Large-study SAT coverage is still pending.

Run reproducible positive/negative/interface controls:

```powershell
.venv/Scripts/python.exe exp/tool-exploration/exp-tool-13-rover/check.py --container codex-sandbox-agent-workspace --controls-only
```

Omit `--controls-only` to run the larger 16/32-bit studies as well. Omit
`--container` when local Yosys is available. Every attempt has a 120 s SAT limit;
a timeout is an inconclusive result, not an accepted proof.
