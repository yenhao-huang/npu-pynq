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

## Pre-optimization source semantics

Development run 14c87d incorrectly accepted `x[x] ^ x[x]` on an 8-bit input:
out-of-range indexing was simplified out before coefficient propagation. This
result is rejected, not accepted equivalence evidence. The affine checker now
composes `yosys_equivalence` self-checks for each source with its coefficient
algorithm. Each self-check requires complete interfaces, total binary source
semantics and matching source hashes before netlist optimization. A self-check
never replaces the cross-design affine proof. Runtime limits apply to each child
proof or synthesis process; a failed or unknown guard returns no affine proof.

Eight actual source controls match their expected verdicts, including cancelled
out-of-range indexing in either source separately. Valid constant division by
one passes. The original CRC64 pair still proves, a different polynomial fails,
and its concrete counterexample replays in Icarus. Nonlinear AND still rejects.
All four distinct historical CRC/LFSR source pairs (eight recorded study proofs)
prove again with the new guards, after validating original hashes. Old studies
and physical measurements remain unchanged; these are additional proof records.

```sh
python exp/tool-exploration/exp-tool-15-gf2/source_controls.py --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-15-gf2/recheck_studies.py --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-15-gf2/study.py
```

See `affine-source-controls.json`, `affine-source-rechecks.json`,
`affine-guarded-controls.json` and `affine-source-development-failure.json` under
`docs/goals/0928-tool-exploration/evidence/`. No new operation is counted.
