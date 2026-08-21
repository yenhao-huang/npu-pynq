## 1. Numeric Contract

- [x] 1.1 Implement signed INT8/INT32 saturation, MAC, matrix multiplication, and integer requantization in `src/test/model/numeric.py`; verify focused endpoint, overflow, rounding, reset-equivalent, layout, and invalid-input unit tests pass.
- [x] 1.2 Add deterministic matrix fixtures and verify dense row-major output matches an independent scalar reference for normal, signed-endpoint, and non-square shapes.

## 2. Hardware ABI

- [x] 2.1 Implement ABI identity, version, capabilities, register offsets, control/status bits, and error enums in `src/test/model/abi.py`; verify exact values and uniqueness with unit tests.
- [x] 2.2 Implement immutable matrix job validation and compatibility negotiation; verify dimensions, strides, timeout, version, capability rejection, and structured error-code cases pass.

## 3. Performance Model

- [x] 3.1 Implement target/configuration records and matrix operation, payload, tiled-cycle, roofline-time, throughput, and limiting-factor calculations in `src/test/model/performance.py`; verify specification examples and edge tiles pass.
- [x] 3.2 Implement resource budget and model-to-measurement acceptance reports; verify boundary percentages and the inclusive ten-percent cycle gate pass.

## 4. Integration and Evidence

- [x] 4.1 Export the contract API from `src/test/model/__init__.py`, add unittest discovery under `src/test/tests/`, and add a Makefile `model` gate; verify `python -m unittest discover -s src/test/tests -v` passes.
- [x] 4.2 Run the repository model, lint, simulation, strict OpenSpec, whitespace, generated-artifact, and secret gates; record exact outcomes and any host-only blockers in `STATE.md`.
- [x] 4.3 Inspect all Phase 0 artifacts for internal consistency, mark this checklist with objective evidence, and prepare an issue-linked Conventional Commit without modifying human-owned documents.
