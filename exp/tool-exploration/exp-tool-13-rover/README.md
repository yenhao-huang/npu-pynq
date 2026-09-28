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

## Bitwise proofs and product abstraction

`normalization=bitwise` proves every output bit independently. The verdict
requires the exact number of successful obligations, exit zero, unchanged source
and no warnings/timeouts. A partial proof never passes. The whole process remains
bounded even though each individual SAT command also has a timeout.

`bitwise_products` first merges identical word cells, replaces each remaining
product with an arbitrary auxiliary input, and then proves all output bits.
This is a stronger overapproximation: products with different inputs remain
independent, so it may reject equivalent circuits. An abstract counterexample
does not establish a concrete RTL bug. It is useful for identical products with
different addition structures; it is generally unsuitable for distributivity.

The original elaborated netlist must contain only total binary word operations
from the explicit allowlist and no X/Z constants. Division, variable part-select
shiftx, ambiguous pmux, symbolic sources and unknown cells are rejected. A pmux
is accepted only when its selectors are distinct equality decodes of one binary
bus and exhaust every value; this makes exactly one data input active. This
prevents defined auxiliary inputs from hiding an undefined original product.
The installed Yosys 0.23 `help sat`, `help cutpoint`, `help expose` and
`help rename` define the primitive commands. `rename -witness` makes cutpoint
wires public before exposing them; omitting it caused initial controls to fail.

Run `python exp/tool-exploration/exp-tool-13-rover/bitwise.py --container
codex-sandbox-agent-workspace` as one command. Positive, highest-bit mutation,
different-product, X-product and divide-by-zero controls exercise both modes.
All ten expected verdicts match; results are in bitwise-proof-controls.json.
No new operation is counted for these proof modes.

## Arithmetic normalization and source-level safety

`normalization=macc` uses installed Yosys `alumacc; opt; techmap; opt` before SAT.
It canonicalizes arithmetic structure; it is not a custom arithmetic proof or a
new operation. A copied-source 128-bit popcount diagnostic collapses both
representations to identical logic and proves in under one CPU second. The
registry retry still requires complete vectors, SAT and physical measurements.

All five modes now inspect **both separately elaborated source netlists** after
`proc -noopt`/flatten and before `prep`, normalization or miter construction.
Checking only the optimized miter is unsafe: matching partial functions can
disappear during optimization. Two failed development controls (56e17b and
99fd8b) exposed this; neither is accepted evidence. A total-binary-cell allowlist
rejects X/Z values, symbolic sources, variable division, shiftx and unknown cells.
Constant nonzero divisors are supported. Existing state/interface checks remain.

The bitwise.py experiment now exercises 45 actual controls: five modes times
nine cases, including mutations, X values, self/variable division, masked X,
out-of-range indexing and valid constant division. Every expected verdict
matches. See normalization-controls.json; failed development attempts remain
in normalization-development-failures.json.

Run `bitwise.py --normalization bitwise_prefix --container
codex-sandbox-agent-workspace` for ten ordered-bit controls. The low-bit mutation
must fail before any prefix assumption, and the high-bit mutation must still
fail after earlier obligations succeed. Missing success markers, process failure,
timeout and partial source semantics remain rejection conditions.

## Complete interface enforcement

Development run `0ccec1` incorrectly accepted a candidate whose extra output `z`
differed from the reference: the miter observed only `x` and `y`. That run is
explicitly rejected in interface-development-failure.json. The backend now
requires both pre-optimization source netlists to contain exactly input `x` and
output `y`, at the declared widths. A successful SAT result cannot override this
check. Extra unused inputs, extra outputs, inout ports and width mismatches fail.

The Docker Yosys 0.23 controls cover all six normalization modes: 48 interface
cases and 60 arithmetic/undefined-value cases match their expected verdicts.
The historical source audit re-elaborated 98 source files from 49 successful
SAT study proofs, after matching each file set to its recorded source hash. All
have complete declared interfaces and pass the total-binary source-netlist
check. This is not a new cross-design SAT proof or retrospective change to a
historical verdict. Its scope is the evidence directory at execution; later
studies require another audit.

Reproduce with `exp/tool-exploration/exp-tool-13-rover/interfaces.py`,
`bitwise.py` and `audit_interfaces.py`, each with
`--container codex-sandbox-agent-workspace`. See interface-controls.json,
normalization-interface-controls.json and historical-interface-audit.json under
`docs/goals/0928-tool-exploration/evidence/`. Operation 13's identified source
contract issues are fixed; historical physical coverage remains incomplete.
