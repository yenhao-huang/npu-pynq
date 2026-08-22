## 1. Processing Element

- [x] 1.1 Add a self-checking PE testbench that covers reset, clear, independent valids, signed endpoints, positive/negative saturation, and enable stalls; verify it initially fails before RTL exists.
- [x] 1.2 Implement the synthesizable signed PE with one-stage forwarding and per-MAC INT32 saturation; verify the focused Icarus PE simulation passes.

## 2. Parameterized Systolic Array

- [x] 2.1 Add deterministic self-checking 2x2 and 2x3 array tests for skew scheduling, row-major result mapping, logical lane masking, job-boundary clear, and mid-job stall; verify they initially fail before array RTL exists.
- [x] 2.2 Implement flattened parameterized rectangular array composition and parameter guards; verify both deterministic Icarus simulations pass against Phase 0 expected results.

## 3. Differential Verification

- [x] 3.1 Implement a seeded vector generator using `src.test.model.matmul_int8`, generate at least 100 tracked 2x2-physical cases with signed endpoints, and verify byte-for-byte reproducibility in focused Python tests.
- [x] 3.2 Add a self-checking randomized SystemVerilog harness that consumes the tracked vectors with deterministic stalls; verify every active accumulator matches golden output and unused lanes remain zero.

## 4. Integration and Evidence

- [x] 4.1 Update simulation entry points only as required for reliable testbench discovery; verify all 36 Phase 0 Python tests plus new vector tests pass.
- [x] 4.2 Run focused direct Icarus PE/array/random simulations, strict OpenSpec validation, whitespace, generated-artifact, secret, and repository lint/simulation gates; record exact pass, blocked, and not-applicable outcomes in `STATE.md`.
- [x] 4.3 Inspect Phase 1A contract parity and complete diff, mark all tasks with objective evidence, and prepare issue-linked Conventional Commits without modifying human-owned documents.
