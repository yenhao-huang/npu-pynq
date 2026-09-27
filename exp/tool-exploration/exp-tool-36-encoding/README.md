# synth_fsm

Paper connection: Low-Power FSMs in FPGA: Encoding Alternatives, PATMOS 2002,
https://doi.org/10.1007/3-540-45716-X_36. Author manuscript indexed at
https://arantxa.ii.uam.es/~ivan/patmos02-enc.pdf.
The study motivates comparing binary and one-hot storage and output decoding.
This operation does not reproduce its low-power methods or report power savings.

Bounded contract: a synchronous cyclic phase controller with 4..256 power-of-two
states. Reset selects phase zero; advance moves one phase with wrap; otherwise
state holds. Output has one active bit per phase. Initial state before reset and
fault-induced illegal states are outside the contract. It is not an arbitrary
transition-graph compiler. Explicit state attributes preserve the encoding
comparison and inhibit shift-register extraction for the one-hot state ring.

The independent integer-state checker is an extension of sequential_scoreboard,
not an extra counted operation. It checks every post-reset cycle, all states,
hold, wrap, resets and two-cycle fixture observations. Mutations break reset,
enable, rotation direction and fixture output. Bounded tests are not formal proof.

Predeclared physical cases are 64 and 128 states, binary versus onehot, area
objective, three matched Basic-flow pairs. Report FF growth separately: one-hot
encoding can trade registers for decode LUTs. No gain is assumed in advance.

```sh
python exp/tool-exploration/exp-tool-36-encoding/study.py --verify-only --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-36-encoding/study.py --container codex-sandbox-agent-workspace
```

The measurement fixture registers inputs and outputs. The declared transition
latency is three cycles including that fixture, with II=1. Its delayed control
observations are checked independently; no board timing or power claim is made.
