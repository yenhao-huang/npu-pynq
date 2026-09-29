# synth_banked_regfile

Primary paper: LaForest and Steffan, Efficient Multi-Ported Memories for FPGAs,
FPGA 2010: https://fpgacpu.ca/publications/FPGA2010-LaForest-Paper.pdf.
The paper discusses conventional logic storage, banking, replication and
multipumping before its LVT architecture. This tool adapts the banking/logic
comparison. It does not implement the paper's LVT or promise arbitrary true
multiport access without conflicts.

Bounded interface: width 8..64, depth 16..256 and banks 2/4/8, all storage
dimensions powers of two. Each command has one write and two optional reads.
Low address bits select a bank. Read0 wins a same-bank conflict; read1 is
rejected with zero data even if both addresses match. Both storage variants
enforce that rule. Reads return old data on a simultaneous write to the same
word. Synchronous reset clears logical contents and output flags. Distributed
RAM retains raw bits and uses per-word validity masking to implement clearing.

Input x packs write data, write address, read0 address, read1 address, write
enable, read0 enable and read1 enable from low to high. Output y packs read0
data, read1 data, accepted0, accepted1 and conflict. Before the first reset edge
outputs are unspecified. Every command produces its response one cycle later.
The registered observation fixture adds two cycles. II=1 means command batches;
it does not promise two completed reads per cycle.

An independent Python memory model checks every output, every address, bank
conflicts, read1-only traffic, read-before-write, reset after nonzero writes and
8192 random stress cycles. Mutations break arbitration, clearing, bank rows,
collision ordering and fixture outputs. This extends operation 48, not its count.

Predeclared physical cases: 16x64/two banks and 32x128/four banks, resettable FF
baseline versus distributed banked candidate, area objective, three Basic-flow
pairs. No resource benefit is assumed before measurement.

```sh
python exp/tool-exploration/exp-tool-35-multiport/study.py --verify-only --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-35-multiport/study.py --container codex-sandbox-agent-workspace
```

Physical reports include the newer clock-coverage artifacts. Bounded sequential
simulation is not unbounded formal proof; board, power and deployment are outside
this experiment.
