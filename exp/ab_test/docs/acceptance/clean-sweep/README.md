# Clean sweeps: multi-round wins on area, power and delay at once

Five cases where a multi-round candidate is **not worse than baseline on any of the
three metrics** and strictly better on at least one, and where the idea behind the win
fits in a sentence. Nothing here trades one metric for another, and nothing here is
flagged for clocking state off an internally generated signal.

This is a different selection from [`../improvement/`](../improvement/), which ranks by
PDAP alone and therefore includes trades. `JC_counter`, the largest headline
improvement, is absent from this folder: it buys area and power with 2.8x the delay.

Files are byte-identical copies of the run records. Measurements are Nangate45,
10 ns clock, input activity 0.1, OpenROAD 26Q1-1744; `delay_ns = 1000 / fmax_mhz`.

| Case | Model | Ratio | Best at | area um2 | power W | delay ns |
|---|---|---:|---|---|---|---|
| [`multi_8bit`](./multi_8bit/) | Opus 5.5 | 0.175x | round 5 | 529.61 -> 373.20 | 9.48e-05 -> 5.09e-05 | 1.640 -> 0.760 |
| [`freq_div`](./freq_div/) | Sonnet 5.5 | 0.245x | round 4 | 98.42 -> 73.15 | 1.11e-05 -> 5.21e-06 | 0.370 -> 0.260 |
| [`edge_detect`](./edge_detect/) | Opus 5.5 | 0.317x | round 4 | 18.35 -> 12.50 | 2.13e-06 -> 1.22e-06 | 0.160 -> 0.130 |
| [`up_down_counter`](./up_down_counter/) | Sonnet 5.5 | 0.330x | round 2 | 177.42 -> 154.28 | 3.71e-05 -> 1.74e-05 | 0.680 -> 0.550 |
| [`fsm`](./fsm/) | Sonnet 5.5 | 0.362x | round 2 | 38.04 -> 26.87 | 4.90e-06 -> 2.51e-06 | 0.180 -> 0.180 |

Ratio is multi-round PDAP over baseline PDAP, `PDAP = power_w x delay_ns x area_um2`.
`fsm` ties on delay; the other four improve all three.

## `fsm` — 0.362x

The specification is a sequence detector written as five named states. The baseline
implements it literally: a 3-bit state register and a `case` block of next-state logic.

Round 2 keeps the last four values of `IN` in a shift register and makes `MATCH` one
equality against `4'b1001`. There is no next-state logic left — the register's next
value is just its own contents shifted by one.

The state space got *larger*, 16 encodings instead of 5, and the hardware got smaller,
because the saving is in the next-state cone, not in the flops. This is the clearest
case in the set of a win that a state-count heuristic would have ruled out.

## `edge_detect` — 0.317x

The baseline registers `a`, then registers `a & ~a_d` and `~a & a_d` — three flops, and
the two outputs come straight off flops.

Round 4 stores `~a` delayed two cycles instead of `a`, which turns `rise` into a plain
AND2 and `down` into a plain NOR2. Two flops instead of three, and the output gates
shrink to one level each.

Note that `rise` and `down` become combinational outputs rather than registered ones.
They carry the same value in the same cycle, and the measured delay still improves
(0.160 -> 0.130 ns) because the remaining logic is a single gate, but a consumer that
relied on glitch-free registered outputs would see a difference.

## `up_down_counter` — 0.330x

The baseline adds `+1` or `-1` through one 16-bit carry-propagate adder.

Round 2 uses the fact that incrementing and decrementing are the same operation on a
sign-folded value: both flip the trailing run of bits. With `x = count ^ {16{~up_down}}`
and `t[i] = &x[i-1:0]`, the update is `count <= count ^ t`. An AND-prefix chain replaces
the adder — half the power and a shorter critical path.

## `freq_div` — 0.245x

The baseline runs three independent counters and registers each divided clock on its own
toggle flop.

Round 4 offsets two 4-bit counters to run 3..12 instead of 0..9. In that range the wrap
condition is `a[3] & a[2]`, two bits and no comparator, and the duty cycle falls out of
the encoding so the divided clocks can be taken directly off counter bits:
`CLK_50 = ~a[0]`, `CLK_10 = a[3]`, `CLK_1 = b[3]`. The three toggle flops disappear.

Choosing the counter's offset so that the outputs are already bits of the state is the
whole optimization.

## `multi_8bit` — 0.175x

The baseline is a behavioral shift-and-add loop, which synthesizes to eight 16-bit adds
in series. That serial carry chain is why its delay is 1.640 ns, by far the worst in
this set.

Round 5 writes the eight partial products explicitly and sums them as a tree: adjacent
pairs first, then one 4:2 carry-save stage, then a single carry-propagate add at the end.
Delay more than halves and area drops 30% at the same time.

The textbook transformation, and the one case here where the win is in a standard
datapath structure rather than in a problem-specific observation.

## Reproduce

Each folder holds `baseline.v` and `multi-round.v`. Both compile and pass the RTLLM
testbench that ships with the problem:

```bash
cd exp/ab_test
iverilog -g2012 -o /tmp/a.vvp docs/acceptance/clean-sweep/fsm/multi-round.v \
    "problems/Control/Finite State Machine/fsm/testbench.v" && vvp /tmp/a.vvp
```

The testbench of each case:

| Case | Problem directory |
|---|---|
| `multi_8bit` | `problems/Arithmetic/Multiplier/multi_8bit/` |
| `freq_div` | `problems/Miscellaneous/Frequency divider/freq_div/` |
| `edge_detect` | `problems/Miscellaneous/Others/edge_detect/` |
| `up_down_counter` | `problems/Control/Counter/up_down_counter/` |
| `fsm` | `problems/Control/Finite State Machine/fsm/` |

Selection method and the geometric means behind it:
[../../multi-round-analysis.md](../../multi-round-analysis.md).
