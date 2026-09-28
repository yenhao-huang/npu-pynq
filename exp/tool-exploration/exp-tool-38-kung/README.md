# synth_systolic_tile

Primary connection: H. T. Kung, *Why Systolic Architectures?*, Computer 15(1),
37-46 (1982), DOI 10.1109/MC.1982.1653825,
https://www.eecs.harvard.edu/~htk/publication/1982-kung-why-systolic-architecture.pdf.
This bounded adaptation uses repeated local MAC cells and regular neighbor
communication. It does not reproduce a historical chip or its performance.

The generator accepts signed 8..32-bit elements and square dimensions 2..4.
Input packs row-major A in low bits, then row-major B. Output packs row-major
C=A*B, with 2*width+ceil(log2(size)) bits per signed result. There is no saturation
or truncation of the mathematical sum. Parallel architecture computes all
products and sums before its output register. Systolic architecture captures
the matrices, injects skewed rows/columns, forwards A right and B down one PE
per cycle, and accumulates each output locally. Last-wavefront capture includes
the final product. This is a real 2D dataflow, not a renamed dot-product helper.

One matrix batch is outstanding. Synchronous reset cancels it; completed output
is held under backpressure. Parallel minimum latency/II are 1/1; systolic values
are (3*size-1)/(3*size-1), including operand capture and wavefront drain. Throughput
counts whole matrix batches. The registered observation fixture adds two cycles
and is not an external handshake adapter.

An independent Python matrix-product oracle uses exact signed integers, without
PE state or injection scheduling. The transaction checker verifies every
ready/valid cycle, output ordering, holds, cancellation and calibrated latency/II.
Directed cases include identity, nonsymmetric matrices, signed extrema and
one-hot routes for every i,k,j product; each substantial case also runs 8192
seeded stress cycles. Nine negative controls break forwarding, skew, signedness,
accumulation, reset, hold, latency, II and fixture observation. These are bounded
sequential tests, not unbounded formal equivalence.

Predeclared physical cases: 2x2 matrices at 16/32-bit element widths, parallel
versus systolic, area objective, three matched Basic-flow pairs on xc7z020.
Both request DSP inference; actual LUT/FF/DSP/BRAM costs must be reported.
Any area savings with excessive throughput loss remain tradeoffs.

```sh
python exp/tool-exploration/exp-tool-38-kung/study.py --verify-only --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-38-kung/study.py --container codex-sandbox-agent-workspace
```

Verification a971e4 passes all eight source-bound core/fixture checks. Pipeline
tests reject changing matrix dimensions across a comparison. Larger 3x3/4x4
configurations have simulator coverage; they are not substituted for the two
declared physical configurations.

`study.py --case w32_n2` reruns only the existing declared width32 case. Retry
514661 preserves the original area objective and all three physical pairs.
The new detailed audit accepts the baseline's bypassed DSPs, but still rejects
four candidate DSPs with CREG=1. LUT reduction 15.04% and throughput loss 82.91%
are diagnostic until coverage is complete. Neither threshold qualifies anyway.
Do not replace the older physical study or edit its coverage reports.
