# Paper survey and extraction decisions

Scope: AI-assisted RTL optimization, prioritizing ICCAD/DAC publications and
related primary papers. Accessed 2026-09-28. The tools are small engineering
adaptations; none claims to reproduce a complete published learning framework.

| Paper | Publication evidence | Method extracted | Local boundary |
| --- | --- | --- | --- |
| RTLRewriter | ICCAD 2024; [primary paper](https://arxiv.org/html/2409.11414v1), DOI 10.1145/3676536.3676775 | Retrieve optimization guidance, reason about operand widths, select cost-aware rewrite actions, verify before synthesis | Curated rules, interval arithmetic, cost-normalized UCB and a gated pipeline; no multimodal retrieval, learned partitioner or full C-MCTS |
| PPA-RTL: Hardware Generation with High Flexibility using Reinforcement Learning Enhanced LLMs | DAC 2025; [author-hosted paper](https://ece.k-state.edu/research/hardware-security/papers/DAC2025_RLPFA_Processing.pdf), DOI 10.1109/DAC63849.2025.11132897 | Guide optimization with configurable PPA preferences | Weighted normalized improvement with correctness and hard-limit gates; no model training or claim to implement the paper's exact reward |
| Enhancing LLMs for HDL Code Optimization using Domain Knowledge Injection (Mascot / RTLOpt) | DAC 2025; [IBM Research publication](https://research.ibm.com/publications/enhancing-llms-for-hdl-code-optimization-using-domain-knowledge-injection) | Use domain knowledge for pipelining and clock-related optimization | Rule advice with latency/reset/FPGA-enable preconditions; no sequential transformation or multi-agent framework |
| ASPEN: LLM-Guided E-Graph Rewriting for RTL Datapath Optimization | MLCAD 2025; [author-hosted paper](https://www.csl.cornell.edu/~zhiruz/pdfs/aspen-mlcad2025.pdf) | Use actual synthesis feedback and retain Pareto trade-offs | Routed FPGA measurements, comparison/provenance and Pareto filtering; no e-graph engine or paper benchmark replication |
| SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning | [Primary preprint, arXiv:2504.10369](https://arxiv.org/html/2504.10369v1) | Rule-guided rewriting and symbolic FSM optimization with verification | FSM rule retrieval with explicit sequential-proof requirements; no symbolic FSM solver |

## Why these operations

The papers expose recurring needs that the existing lint/sim/debug/synth tools
do not cover: choosing a rewrite, explaining its assumptions, interpreting
measurements consistently, retaining trade-offs and deciding where to spend the
next EDA run. Those operations can be useful to any agent without coupling the
repository to one model provider, training method or paper's private toolchain.

The adder-sharing experiment is an original four-bit instance of the arithmetic
sharing pattern discussed by RTLRewriter. It uses no downloaded benchmark RTL.
Width advice is our exact interval-arithmetic utility, motivated by the paper's
width-aware program analysis; it is not the paper's synthesis-time predictor.

## Rejected scope for this delivery

Full reinforcement learning needs a model, data, reward calibration and training
budget. E-graph equality saturation needs a typed RTL IR and rewrite proof
infrastructure. Sequential state merging/pipelining needs cycle/reset-aware
verification. These are worthwhile future work but are not disguised as finished
features here. The combinational pipeline is intentionally bounded and records
that scope in every result. FPGA LUT/delay results do not establish ASIC PPA or
NPU system performance. Paper-reported percentages are not used as local metrics.

## Expanded arithmetic and physical-feedback foundations

[ROVER (TCAD 2024)](https://arxiv.org/html/2406.12421v1) explores arithmetic
rewrites across widths and signedness, extracts implementations using a cost
model, and checks transformations. The expanded tools borrow its separation of
rewrite construction, correctness and cost; they do not implement its e-graph
engine. Serial, balanced and carry-save reduction are one generator operation
with three alternatives, not three entries in the tool count.

The new `yosys_equivalence` operation uses a flattened combinational miter and
Yosys SAT. `vector_equivalence` adds reproducible large-interface diagnostics;
it cannot replace the proof. `clocked_ppa` extends physical-feedback evaluation
to registered OOC boundaries, resource-class accounting and II-normalized
throughput. Its timing estimate is scoped to register-to-register setup paths.

[PrefixLLM](https://arxiv.org/html/2412.02594v1) and
[PrefixRL](https://arxiv.org/abs/2205.07000) motivate structured prefix-network
exploration with synthesis feedback. Prefix generators and their experiments
remain planned work, not delivered tools.

## Diverse network and storage studies

[PrefixRL (DAC 2021)](https://arxiv.org/abs/2205.07000) synthesizes prefix
circuits, including adders and priority encoders, with physical feedback. The
priority and leading-zero generators use explicit hierarchical networks to
expose topology choices. They do not reproduce its reinforcement-learning agent.

Popcount uses narrowing-aware reduction (ROVER-inspired arithmetic exploration).
Barrel shift, masked selection and stable argmax are additional RTLRewriter-style
rewrite subjects with explicit total semantics. These are hand-authored
architecture alternatives used to exercise correctness and physical-feedback
tools; no claim is made that the papers present those exact generators.

[Scalar Replacement with Circular Buffers (Seto, 2019)](https://www.jstage.jst.go.jp/article/ipsjtsldm/12/0/12_13/_article)
substitutes RAM-based circular storage for costly shift-register chains in HLS.
The FIFO generator adapts that storage choice to ready/valid buffering. It
preserves order, capacity and reset behavior; it does not implement the paper's
compiler pass. An independent queue scoreboard checks temporal correctness
before physical comparison. Large-storage timing needs a registered fixture
covering read and handshake paths, which remains pending.

## Arithmetic and paired physical evidence

The prefix generator implements explicit Kogge-Stone and Sklansky networks,
using PrefixLLM's structured-topology idea without an LLM search loop. CSD and
MCM expose shift/add graph choices motivated by ROVER. The shared MCM generator
memoizes factors 2^k +/- 1; it is a bounded heuristic, not an optimal adder-graph
solver. All numeric contracts are unsigned and retain the full product width.

Physical-analysis tools validate clock/II consistency and separate LUT, FF,
DSP and BRAM changes. Repeated results require distinct report handles and
unchanged source/build/flow identities. These are engineering checks around
ASPEN-style physical feedback, not extra claimed reproductions of its algorithm.

## Structural diagnostics and constrained storage

Yosys generic and xc7 netlists make the feedback loop inspectable: cell/operator
attribution, fan-in depth and pin fanout identify concrete rewrite consequences,
while memory classification detects storage substitutions. These diagnostics
are engineering adaptations around the RTLRewriter/ASPEN feedback loop.
The circular-storage comparison remains tied to Seto's scalar-replacement
method. A distributed-RAM policy is an explicit target-resource constraint,
not a claim that memory bits disappeared. The existing 0-BRAM baseline budget
must be preserved for an unconditional LUT-area win.

## Constant residue arithmetic (operation 32)

Campbell, Lin and Chen, *Low-cost hardware architectures for mersenne modulo
functional units*, ASP-DAC 2018, pp. 599-604, DOI 10.1109/ASPDAC.2018.8297388.
Primary institutional record:
https://experts.illinois.edu/en/publications/low-cost-hardware-architectures-for-mersenne-modulo-functional-un/

The paper motivates inexpensive modulo shadow datapaths for arithmetic error
checking. Our bounded adaptation emits a binary balanced chunk-sum and
end-around-fold reducer, with canonical zero and conservative intermediate
bounds. It is not a reproduction of the paper's full-adder-only standard-cell
architecture, residue encoding, multiplier or equality unit. Local PPA results
must come from the declared 16/32-bit FPGA experiments, not the paper's results.

## Signed arithmetic datapaths

The FIR-window and dot-product generators connect to Thomas B. Preusser,
*Generic and Universal Parallel Matrix Summation with a Flexible Compression
Goal for Xilinx FPGAs*, FPL 2017, DOI 10.23919/FPL.2017.8056834.
Primary author manuscript: https://arxiv.org/html/1806.08095v1.
The paper explains the 3:2 full-adder value invariant and matrix compression for
multiply-accumulate and filter kernels. Our dot product uses ordinary 3:2 rows;
it does not reproduce its Xilinx counter-selection heuristic. FIR symmetry uses
binary preaddition plus balanced accumulation. AMD's symmetric FIR documentation
is supplementary implementation context, not an additional paper reproduction.
Saturation sharing adapts ROVER's equivalence-guided arithmetic resource rewriting
with a single conditional-inversion adder; no e-graph search engine is claimed.

## CRC, jump and exact linear verification

Zhang, Wu, Zhou and Chen, *An optimized delay-aware common subexpression
elimination algorithm for hardware implementation of binary-field linear
transform*, IEICE Electronics Express 11(22), 2014, DOI 10.1587/elex.11.20140934.
Primary PDF: https://www.jstage.jst.go.jp/article/elex/11/22/11_11.20140934/_pdf.
We adapt repeated XOR-pair extraction; the paper's delay-aware gate is not
implemented. Matrix equality motivates the independent exact affine checker.

Haramoto et al., *Efficient Jump Ahead for F2-Linear Random Number Generators*,
GERAD G-2006-62: https://www.gerad.ca/en/papers/G-2006-62.
Our hardware generator uses the binary-matrix baseline, not the report's faster
polynomial sliding-window technique. No PRNG-quality or maximum-period claim.

Parallel CRC context: Gangopadhyay and Reyhani-Masoleh, DOI
10.1109/TC.2015.2479617, author manuscript at
https://www.eng.uwo.ca/electrical/faculty/reyhani_a/docs/publications/GRM-TC-16.pdf.
The paper's concurrent parity fault-detection architecture is not reproduced.

## Iterative arithmetic and verified cycle cost

Aggoun, Farwan, Ibrahim and Ashur, *Radix-2^n Serial-Serial Multipliers*,
author manuscript: https://bura.brunel.ac.uk/bitstream/2438/2756/3/Paper%202438-2756.PDF.
The paper motivates digit-size and resource-reuse exploration. Our adaptation
accepts full words with ready/valid and folds shift/add work over one or two bits
per working cycle; it does not reproduce the paper's serial-serial pin interface,
dependency-graph projection or sub-digit pipeline.

Sutter and Deschamps, *High Speed Fixed Point Dividers for FPGAs*, FPL 2009,
author companion: https://www.arithmetic-circuits.org/FixedPointDivision/FixedPointDivision.html.
Its author-indexed companion lists digit-recurrence architectures and multiple
radices. Direct page retrieval was unavailable during this checkpoint; no claim
about its internal cells or measured results is made. Our bounded exploration
uses conventional exact restoring steps, one or two per cycle, and a parallel
division baseline. The two-step variant is not an SRT implementation.

Operation 44 connects RTLRewriter's verification-before-evaluation workflow to
an independent sequential transaction oracle. It verifies first-valid latency
and unstalled initiation interval from full ready/valid traces before exposing
cycle metrics. It is not an unbounded sequential proof or a paper reproduction.

## Signed Booth recoding

Andrew D. Booth, *A Signed Binary Multiplication Technique*, QJMAM 4(2),
236-240 (1951), DOI https://doi.org/10.1093/qjmam/4.2.236.
Original paper: https://www.ece.ucdavis.edu/~bbaas/281/papers/Booth.1951.pdf.
Operation 21 uses adjacent-bit signed recoding and a paired-bit adaptation,
with balanced full-width sums. The paired circuit is not claimed as a direct
reproduction of the 1951 architecture. Its result remains a full signed product;
no approximate arithmetic or paper-derived FPGA performance claim is used.

## State encoding and phase control

*Low-Power FSMs in FPGA: Encoding Alternatives*, PATMOS 2002,
https://doi.org/10.1007/3-540-45716-X_36. The publisher abstract describes
binary/one-hot and alternative state encodings; its indexed author manuscript
is https://arantxa.ii.uam.es/~ivan/patmos02-enc.pdf. Direct manuscript retrieval
was unavailable. Our bounded cyclic-phase comparison studies the storage and
output-decode tradeoff only. It does not reproduce the paper's power methodology
or claim a measured power benefit. Explicit encoding and no-SRL attributes keep
the intended state-storage alternatives visible to synthesis.

## Constraint coverage and physical evaluation

Operation 43 extends RTLRewriter's controlled PPA-evaluation workflow with an
independent report audit: https://arxiv.org/html/2409.11414v1. No paper-derived
constraint inference is claimed. Installed Vivado 2026.1 `help check_timing`
and `help all_registers`, plus a real generated report, define the six internal
coverage checks and sequential-cell enumeration (including DSP/BRAM registers).
The experiment demonstrates actual rejection of an unclocked sequential domain.

## Banked register storage

LaForest and Steffan, *Efficient Multi-Ported Memories for FPGAs*, FPGA 2010,
https://fpgacpu.ca/publications/FPGA2010-LaForest-Paper.pdf.
The paper contrasts logic storage with conventional banking/replication and
multipumping, then introduces an LVT architecture. Operation 35 adapts the
conventional banking comparison with an explicit read-conflict policy and
distributed RAM. It does not reproduce LVT, unrestricted multiwrite operation,
the Altera implementation or the paper's performance figures. Both baseline
and candidate enforce the same externally visible conflict semantics.


## Stallable stream channels

Carloni, McMillan and Sangiovanni-Vincentelli, *Theory of Latency-Insensitive
Design*, IEEE TCAD 20(9), 1059-1076 (2001), DOI 10.1109/43.945302,
https://www.cs.columbia.edu/~luca/research/lipTransactions.pdf.
The paper develops stallable channels and relay stations. Operation 37 is a
bounded ready/valid buffering adaptation with independent ordering checks.
It does not reproduce the full patient-process framework or establish arbitrary
network deadlock freedom. The experiment exposes storage capacity and full-stage
recovery costs alongside any timing improvement.


## Controlled component attribution

RTLRewriter section 4.3, https://arxiv.org/html/2409.11414v1, compares framework
components through ablation. Operation 47 adapts this controlled-comparison
principle to measured RTL configurations. Complete binary factorial finite
differences extend the paper connection; they are not a claimed implementation
of its partitioning, retrieval or search algorithm. Independent source-bound
correctness remains required before interpreting PPA attribution.


## Spatial matrix computation

H. T. Kung, *Why Systolic Architectures?*, Computer 15(1), 37-46 (1982),
DOI 10.1109/MC.1982.1653825,
https://www.eecs.harvard.edu/~htk/publication/1982-kung-why-systolic-architecture.pdf.
Operation 38 adapts repeated local MAC cells and neighbor operand forwarding to
small signed matrix batches. A skewed 2D wavefront exposes spatial reuse and
explicit fill/drain latency. The parallel baseline and systolic implementation
use the same complete matrix semantics; no historical chip replication or paper
performance result is claimed.
