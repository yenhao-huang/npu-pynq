# FP16 adder experiment contract

Target: the binary16 adder in RTLScout Section 7.1 (`fpadd_f16`), locally named
`fpadd_fp16`. Ports a, b, y are 16-bit combinational signals (E5M10, bias 15).
Round to nearest, ties to even; gradual underflow; infinity on overflow.
Opposite infinities and all NaN inputs return canonical positive quiet NaN
0x7e00. Exact cancellation produces +0; -0 plus -0 produces -0. There are no
exception flag ports. NaN payload propagation and signaling flags are not
claimed. These interface choices must be distinguished from full IEEE-754
exception behavior and any unpublished benchmark details.

Use the existing pinned Docker OpenROAD and ASAP7 RVT TT NLDM library. Timing
uses a virtual clock, zero input/output delays, 10 ps input transition,
3.898 fF output load, and no interconnect parasitics. Compare post-mapping
cell area and maximum combinational delay; power is a vectorless estimate.

Paper Section 7.1: baseline 58 um^2 / 1610 ps; optimized extremes 49 um^2 and
1043 ps, not assumed to occur at the same point. The exact paper flow is not
reproduced by matching width alone. Report all tool/constraint differences.

Verification: Python exact integers cross-checked against binary64 addition
followed by binary16 packing; C++ exact integer oracle checks RTL and gates.
Standard screening covers all encodings against directed anchors in both
operand orders, near cancellation, and one million seeded random pairs.
Exhaustive mode means all 4,294,967,296 pairs, never the FP12 input domain.

## Iteration record

- 000_exact_baseline: establish correctness using exact 42-bit fixed-point
  addition before normalization and rounding. Expected to be expensive;
  establishes a transparent baseline for compact alignment architectures.
  Command: `python run_eval.py --name 000_exact_baseline --delay 1600 --constrained`.
  Result and next decision are pending the recorded run output.

## Baseline result and next steps

000_exact_baseline passed 3,359,296 RTL and 3,359,296 gate checks:
125.44632 um^2, 2110.819092 ps. The compact alignment design reduced cost
substantially; see `report.md` for all current results. Run 001 failed to
compile because `small` is a SystemVerilog keyword; run 002 renamed the
signal and passed. Failed run 001 remains recorded and is not counted.

The second round (`refine.py`) tests parallel exponent differences and removes
ABC driving/load constraints while retaining identical final STA constraints.
This distinguishes mapping heuristic effects from measurement changes.
