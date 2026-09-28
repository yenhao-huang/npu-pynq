# Tool 19: symmetric FIR arithmetic

Hypothesis: coefficient symmetry halves the independent constant-product nodes
by preadding mirrored samples at one extra bit before multiplication. Both forms
use balanced final accumulation so the comparison isolates symmetry. A third
`direct_serial` variant supports a separate accumulation-order ablation.

This is a signed FIR-window arithmetic kernel; the caller supplies sample
history. It is not a streaming FIR with a delay line or a ready/valid protocol.
Coefficients are signed constants. The output width accommodates the entire
sum of absolute coefficient magnitudes, without saturation or rounding.

Paper connection: Preusser, *Generic and Universal Parallel Matrix Summation
with a Flexible Compression Goal for Xilinx FPGAs*, FPL 2017,
[author manuscript](https://arxiv.org/abs/1806.08095), DOI 10.23919/FPL.2017.8056834.
The paper connects multioperand sum implementations to FIR/dot-product kernels.
Symmetry preaddition is a bounded algebraic adaptation, not a reconstruction of
its device-specific counter heuristic. AMD's
[symmetric FIR documentation](https://docs.amd.com/r/en-US/am004-versal-dsp-engine/Symmetric-Systolic-FIR-Filter)
provides the complementary hardware rationale; this project targets xc7z020,
not the manual's DSP58 device.

Predeclared PPA: signed 16/32-bit samples, 16 taps, symmetric positive coefficients,
matched logic-only multiplier policy, area objective, 5 ns Basic flow, three pairs.
There are 16/8 constant product nodes before synthesis, not measured resource counts.
Independent integer tests also cover negative/zero coefficients, odd tap counts,
minimum samples and maximum coefficients. Broken preaddition must fail checking.

Run `study.py --verify-only --container codex-sandbox-agent-workspace` first;
omit `--verify-only` for physical execution after proof. Preserve all SAT timeouts
and failed implementations. A generator node count never establishes a PPA gain.

Use `study.py --configuration config-prefix.json --container
codex-sandbox-agent-workspace --verify-only` for a 300-second ordered-bit retry
per original configuration. This retry preserves the exact original cases,
architectures and area objectives. Its outcome remains pending at this checkpoint;
partial obligations and timeouts cannot unlock physical acceptance.

## Bounded proof normalization attempts

Run `study.py --configuration config-macc.json --verify-only --container
codex-sandbox-agent-workspace`, or substitute `config-aig.json`, using this
directory's study.py path from the repository root. Each keeps the original
width16/32 designs and area objectives with a 300-second formal budget.
Macc study 09fb37 and AIG study 08c404 both time out on both cases. Earlier
prefix study 183558 proves only 3/24 and 3/40 obligations. No attempt establishes
complete equivalence or permits a physical acceptance claim.
