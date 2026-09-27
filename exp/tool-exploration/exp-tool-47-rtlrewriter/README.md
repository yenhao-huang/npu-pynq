# candidate_ablation

Primary connection: RTLRewriter, ICCAD 2024, section 4.3,
https://arxiv.org/html/2409.11414v1. The paper ablates framework components.
This operation adapts controlled component comparison to source-bound physical
architectures. Its complete factorial contrasts are a conventional analysis
extension, not the paper's algorithm or a reproduction of its results.

One to four binary configuration factors require all combinations and at least
three executions per measured combination. Files/top bind records to current
source bytes; nonfactor configuration, physical context and per-cell cycle
contracts must match. Duplicate executions, changed files, missing combinations
and unequal repeats are rejected. Explicit failed combinations are retained and
prevent estimation. Input labels and report authenticity still need independent
audit. Correctness is separate; the experiment checks source-bound core/fixture
evidence before analysis.

Outputs include conditional high-minus-low effects and averaged finite
differences through the highest interaction order, in native resource,
throughput, latency and II units. Zero LUTs need no pseudocount. Min/max and
individual repeats describe reproducibility, not confidence intervals. These
effects are distinct from paired percentage gain gates.

Run `python exp/tool-exploration/exp-tool-47-rtlrewriter/analyze.py` to compare
circular FIFO auto/distributed storage at 16x64 and 32x128 using existing
physical evidence. This is retrospective analysis and cannot revise objectives.
Shift storage rejects distributed policy, so it cannot form a full architecture
by storage-policy factorial design. Do not invent that missing cell.

The supplemental study fixes width16 and circular architecture, compares
auto/distributed storage at depth128, and predeclares an area objective with
three Basic-flow pairs. Run `study.py --container codex-sandbox-agent-workspace`
in this directory (or use its repository-relative path). Add `--verify-only`
for correctness only. Supply the completed study JSON to `analyze.py
--supplement PATH` for the full storage-by-depth64/128 factorial analysis.
Depth is a workload factor: its resource effect and interaction quantify scale,
not an optimization win. The new case remains in the all-case denominator.

Historical FIFO reports lack the later timing-coverage audit. The analysis does
not upgrade them to audited timing closure or prove statistical causation.
