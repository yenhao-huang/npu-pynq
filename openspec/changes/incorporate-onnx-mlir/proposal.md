# Proposal

## Why

SmolLM2 export currently compiles NPU work but forwards several CPU operators
to ONNX Runtime through a separately built C object. Integrating ONNX-MLIR
should make the graph available as MLIR before partition and remove the
production ORT dependency after numerical and board validation.

## What Changes

- Add reproducible ONNX-MLIR discovery, source inventory, import, and diagnostics.
- Adapt imported computations to the existing NPU partition input contract.
- Compile remaining CPU computations to ARM32 without ORT session execution.
- Preserve the NPU compiler, ISA, driver, RTL, and package entry-point contracts.
- Remove ORT production assets and manual bridge build steps only after parity.

## Capabilities

### New Capabilities

- `onnx-aot-compilation`: inspect and compile ONNX graphs through an explicit
  MLIR boundary and package CPU/NPU execution without production ORT.

### Modified Capabilities

None. No existing main specifications are registered in this checkout.

## Impact

Related to #119, already claimed by agent `a` in this worktree. The approved
design is `docs/plans/onnx-mlir-incoporate/plan.md`. Changes affect export,
compiler tooling, focused tests, runtime packaging, and example documentation.
Initial compatibility work stays in the existing #119 integration worktree;
subsequent independently mergeable implementation units must follow the
repository's issue/sub-issue workflow before product integration.
