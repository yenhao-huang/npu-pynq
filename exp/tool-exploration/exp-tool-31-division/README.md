# synth_divider

Sutter and Deschamps, High Speed Fixed Point Dividers for FPGAs, FPL 2009. Author companion: https://www.arithmetic-circuits.org/FixedPointDivision/FixedPointDivision.html

This bounded adaptation explores digit size and resource reuse. It does not
reproduce the paper's complete cell architecture or its reported gains.
Unsigned operands are 16 and 32 bits, packed a low / b high. Multiplication
returns the full product. Division returns quotient low and remainder high;
division by zero returns an all-ones quotient and the original dividend.

Parallel, one-bit serial and two-bit-per-cycle variants share a synchronous
reset and single-outstanding ready/valid contract. Reset cancels pending work.
Outputs hold under backpressure. Serial latency/II is width+1; the two-bit
variant is width/2+1. The setup cycle is counted. Parallel latency/II is one.

The predeclared four-case study retains parallel-to-serial area tradeoffs and
serial-to-two-bit throughput tradeoffs. Throughput uses Fmax divided by verified
II. A smaller iterative circuit with a severe throughput regression cannot
qualify as an unconditional area win. Three matched physical pairs per case.
The timing fixture delays observations by two cycles and is not an external
ready/valid adapter. Core and fixture have independent integer-oracle traces.

```sh
python exp/tool-exploration/exp-tool-31-division/study.py --verify-only --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-31-division/study.py --container codex-sandbox-agent-workspace
```

No unbounded sequential formal proof is claimed. Directed extrema, zero divisor,
8192 random stress cycles, reset cancellation, held outputs and sustained
acceptance spacing are required before physical runs. Negative controls are in
test_iterative_exploration.py. Generated RTL and reports stay in ignored stores.
