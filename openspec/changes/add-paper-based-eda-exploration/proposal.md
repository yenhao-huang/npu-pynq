# Paper-based EDA tool exploration

## Why
The current IC tools execute individual EDA steps but do not help an agent
choose optimization rules, compare compatible measurements, or retain Pareto
trade-offs. Issue #80 requests ten paper-informed tools and reproducible evidence.

## What changes
Add eight PPA-oriented operations, a bounded combinational checker, and a
verification-gated measurement pipeline. Use the existing registry and run store.
Publish modular experiments, a paper survey, results, and a reusable skill.

## Non-goals
No retraining an LLM, reproducing entire published frameworks, NPU RTL changes,
ASIC signoff, board deployment, or fabricated performance claims.
