# Change: Deliver the Phase 1C matrix multiplication example

## Why

Phase 1B exposes one bounded physical matrix job, but users still need a public,
repeatable example that multiplies logical matrices through the repository
runtime. Issue #5 requires normal, repeated, and non-tile-aligned cases to use
the same validated DMA path and to compare every result with NumPy.

## What Changes

- Add a runtime-level logical matrix multiplier that tiles arbitrary positive
  M and N dimensions over the discovered physical array dimensions.
- Preserve the signed INT8/INT32 numeric contract and require K to fit the
  hardware `MAX_K` until a separately specified exact K-tiling contract exists.
- Add deterministic host tests for validation, edge tiles, deadline handling,
  repeated execution, result assembly, and performance accounting.
- Add `examples/matrix_multiplication.ipynb` as a thin public consumer of the
  repository runtime, including NumPy comparison and board evidence output.

## Impact

- Affected code: `src/runtime/`, `src/test/tests/`, and `examples/`.
- No RTL, ABI, register-map, Vivado, exporter, or human-owned documentation
  changes.
- Physical board performance remains an explicit acceptance gate and cannot be
  replaced by host tests or simulation.

## Dependencies

- Issue #4 and draft PR #17 provide the runtime and overlay used by this change.
- Issue #5 remains blocked for final acceptance until the PYNQ-Z1 is reachable.
