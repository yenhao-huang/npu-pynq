## 1. Contract and failing tests

- [x] 1.1 Add failing tests for public exports, invalid type/rank/dtype/shape/K/timeout inputs, and no-submit-on-validation-failure.
- [x] 1.2 Add failing tests for normal, edge-tiled 3x5-by-5x3, repeated, non-contiguous input, result ownership, logical deadline, and metrics behavior.
- [x] 1.3 Add failing notebook-structure tests for valid output-free JSON, public runtime imports, NumPy assertions, normal/edge/repeated cases, and absence of direct MMIO/DMA/allocation code.

## 2. Runtime implementation

- [x] 2.1 Implement immutable result/metrics types and a dependency-injected logical matrix multiplier under `src/runtime/`.
- [x] 2.2 Implement M/N edge tiling, dense tile normalization, exact INT32 assembly, one logical deadline, and deterministic performance accounting.
- [x] 2.3 Export the public Phase 1C API and verify all focused host tests pass.

## 3. Public example

- [x] 3.1 Add `examples/matrix_multiplication.ipynb` as a thin public-runtime consumer with normal, non-tile-aligned, and repeated NumPy comparisons.
- [x] 3.2 Add non-secret board performance/provenance output and keep the notebook free of saved outputs and direct hardware control.

## 4. Validation and handoff

- [x] 4.1 Run all Python tests, seven direct Icarus testbenches, strict OpenSpec validation, notebook JSON/source checks, whitespace/generated-artifact/secret/human-document scans, and record evidence in `STATE.md`.
- [x] 4.2 Publish an Issue #5-linked Conventional Commit and draft PR without rewriting Issue #4 history; keep CI and dependency gates explicit.
- [ ] 4.3 Execute the notebook on PYNQ-Z1 with matching Issue #4 artifacts and record exact result/performance evidence; keep this task incomplete while the board or Issue #4 is blocked.
