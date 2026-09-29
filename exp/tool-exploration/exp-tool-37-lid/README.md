# synth_skid_buffer

Primary paper: Carloni, McMillan and Sangiovanni-Vincentelli, Theory of
Latency-Insensitive Design, IEEE TCAD 20(9), 1059-1076 (2001),
https://www.cs.columbia.edu/~luca/research/lipTransactions.pdf.
The paper motivates stallable channels and relay buffering. This bounded
ready/valid adaptation is not a reproduction of the complete patient-process
theory or a proof that arbitrary composed networks are deadlock-free.

The generator builds 2..32 forwarding stages with 8..64-bit payloads. Elastic
stages have one slot and propagate ready combinationally. Skid stages have two
slots; input ready depends only on local registered occupancy, cutting the
downstream ready chain. A full skid stage needs a recovery cycle before accepting
again. Both preserve accepted transaction order and held outputs. Reset flushes
pending transactions. Capacity is stages versus 2*stages and is reported.

The independent model uses logical queues per stage plus an end-to-end queue.
It checks every ready/valid cycle, data order, stall holds, simultaneous transfer,
full capacity and reset. A sustained no-stall phase verifies latency=stages and
II=1. Random/backpressured latency is variable. The fixture adds two observation
cycles and is not an external handshake adapter. Five negative controls break
reset, spill forwarding, capacity, declared architecture and fixture output.

Predeclared cases: width32/stages8 and width64/stages16, throughput objective,
three matched Basic-flow physical pairs. Extra capacity and FF/LUT cost remain
visible and can disqualify an unconditional gain even when Fmax improves.

```sh
python exp/tool-exploration/exp-tool-37-lid/study.py --verify-only --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-37-lid/study.py --container codex-sandbox-agent-workspace
```

Every core/fixture check includes 8192 random stress cycles plus long directed
fill/stall/recovery and calibration phases. Bounded simulation is not unbounded
sequential formal proof. The two architectures are one counted operation.
