# resource_tradeoff

Analyze the actual completed priority-encoder study with this operation. The
shared analysis entry point is exp/tool-exploration/physical_analysis_study.py.
The full study source is preserved in evidence/priority-physical-study.json.

PPA gates use LUT, FF, DSP and BRAM separately, with II-normalized throughput.
Each repeated role must keep its source, timing contract, tool build, device,
clock, directive and implementation-thread setting. Duplicate report handles
cannot count as repeats. Every pair must meet the threshold; results describe
reproducibility, not independent statistical confidence. Correctness and global
family/diversity acceptance remain separate audits. Synthetic negative cases
are in tools/ic/tests/test_physical_analysis.py and never count as physical data.

ASPEN motivates physical-feedback comparison; these audit operations are local
engineering adaptations, not a reproduction of its search policy.
