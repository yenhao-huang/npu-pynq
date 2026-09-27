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
