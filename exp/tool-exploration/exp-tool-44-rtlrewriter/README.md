# latency_throughput

RTLRewriter (ICCAD 2024), https://arxiv.org/html/2409.11414v1,
motivates verification before PPA evaluation. This operation extends that
workflow with explicit cycle contracts for multicycle arithmetic; it is not a
reproduction of RTLRewriter's combinational equivalence checker.

Run `python exp/tool-exploration/exp-tool-44-rtlrewriter/run.py`.
Every 16/32-bit multiplier and divider architecture is checked against Python
integer arithmetic with reset, input holding, output backpressure and a long
unstalled phase. Core and registered observations are both checked. An altered
result bit, wrong zero-divisor result, wrong reset, dropped held output, wrong
latency and wrong II must fail. Cycle metrics are returned only after the entire
trace matches. Bounded simulation does not establish unbounded formal proof.
The paired studies for operations 22 and 31 consume these verified cycle values.
