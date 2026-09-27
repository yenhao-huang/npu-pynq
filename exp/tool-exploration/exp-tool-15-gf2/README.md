# Tool 15 replacement: exact affine equivalence

The proposed separate compressor tool would duplicate an architecture variant
of operation 14. It is replaced by gf2_equivalence, which has an independent
engineering purpose and algorithm: complete symbolic affine equivalence of
source-bound Yosys netlists. It complements operation 13's general SAT checker.

The binary-matrix formulation in Zhang et al., DOI 10.1587/elex.11.20140934,
[primary paper](https://www.jstage.jst.go.jp/article/elex/11/22/11_11.20140934/_pdf),
motivates checking every output coefficient rather than relying on sampled basis
vectors. This checker is our engineering adaptation, not a paper theorem prover.

Contract: complete packed x/y ports, at most 4096 bits each, at most 50000 cells,
no surviving state, unknown bits, undriven nets, cycles or multiple drivers.
Supported cells are XOR/XNOR, inversion, identity, parity reductions and bitwise
AND/OR only when constant-masked or identical expressions. Every accepted output
is an exact input coefficient vector plus one constant bit. Signed/unsigned
extension is explicit. Dependency scheduling is topological, not repeated scanning.
Unsupported nonlinear cells are rejected, even when two supplied sources match.

The checker synthesizes both sources, validates source and netlist digests, and
writes a digest-bound coefficient certificate. A difference produces an exact
binary counterexample. Generation-time matrices are not inputs to the checker.
This proves a two-state relation over all inputs and is not a SAT-success claim.

Run `study.py` on this host with its existing Yosys container. The positive CRC
pair proves, a wrong polynomial fails, its counterexample replays in Icarus, and
a nonlinear AND is rejected. Unit tests additionally cover cycles, missing wires,
state, signed extension, unknowns, duplicate drivers and inverted parity.
