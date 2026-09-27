# Tool 29: parallel CRC transforms

Hypothesis: symbolic expansion and shared XOR pairs replace a deep unrolled
bit recurrence with a smaller parallel network. Inputs are an explicit state
(low word) and data (high word), consumed MSB first. There is no reflection,
implicit initial/final XOR or augmentation. State width 8..64, data width 8..256,
and an odd polynomial below 2**width are required. Primitive polynomials and
standard protocol presets are not assumed.

Paper connection: Zhang et al., *An optimized delay-aware common subexpression
elimination algorithm for hardware implementation of binary-field linear
transform*, IEICE Electronics Express 11(22), 2014,
[primary PDF](https://www.jstage.jst.go.jp/article/elex/11/22/11_11.20140934/_pdf),
DOI 10.1587/elex.11.20140934. We adapt repeated XOR-pair extraction, but do not
implement its delay-aware acceptance heuristic; sharing can worsen timing.
Parallel CRC matrix context is also described in Gangopadhyay and Reyhani-Masoleh,
[author manuscript](https://www.eng.uwo.ca/electrical/faculty/reyhani_a/docs/publications/GRM-TC-16.pdf),
DOI 10.1109/TC.2015.2479617. Its parity fault detector is not reproduced.

Predeclared cases: CRC-32 polynomial 0x04C11DB7, 64/128 data bits, unrolled versus
shared matrix, area objective, Basic flow, 5 ns, three pairs, two attempts.
Exact source-bound affine proofs complement attempted 30-second SAT cross-checks.
All cases retain 8192 random vectors plus boundary patterns. The initial 60-second
128-bit unrolled simulation timed out; the recorded revision raises its bounded
limit to 300 seconds without changing vector coverage or architectures.

Run `study.py --verify-only --container codex-sandbox-agent-workspace`; omit
`--verify-only` for gated physical execution. Reject nonlinear or unknown netlists.
Record actual resource and timing tradeoffs; XOR counts are not FPGA area.
