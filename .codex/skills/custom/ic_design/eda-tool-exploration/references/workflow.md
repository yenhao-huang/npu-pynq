# Tool map and experimental notes

The initial ten operations are optimization_rules, width_advice, ppa_measure,
ppa_provenance, ppa_compare, ppa_pareto, ppa_reward, ppa_select, comb_check and
rtl_evaluate. The first eight support PPA exploration directly. The final two
provide correctness screening and orchestration.

Paper connections: RTLRewriter informs rule retrieval, width-aware reasoning,
cost-aware action selection and verification before synthesis. ASPEN motivates
measured multiobjective exploration and retaining the Pareto frontier. PPA-RTL
motivates user-weighted objectives. SymRTLO and Mascot inform rule preconditions
and sequential optimization examples. These are narrow adaptations, not complete
implementations of the papers' LLM training, e-graphs, symbolic FSM or MCTS systems.

The initial case shares a four-bit adder across a mux. All 8192 binary inputs
are compared. Synthesis can already optimize this form, so zero improvement is
an acceptable observation. Do not add preservation attributes to inflate a
baseline. Check negative controls and measure both designs under identical
Vivado versions, target parts and constraints. Power is not inferred from LUTs.

For new experiments, choose representative RTL and record why the design matters.
Do not extrapolate this small case to the NPU or to a paper benchmark suite.
PPA comparisons need source hashes; correctness evidence must match those hashes.
External Record objects are declarative data, not authenticated reports.
