# Design

## Context

See proposal.md and `docs/plans/onnx-mlir-incoporate/plan.md`. Before migration,
the frontend emitted linalg tensor operations and external ORT calls. The
unchanged NPU pass requires static INT8 linalg contractions whose right operand
is `npu.weight`. The verified implementation uses a pinned LLVM 24 ONNX-MLIR
importer, generic IR parsed by an export-side MLIR 22 adapter, and the existing
LLVM 22 ARM-enabled code generator.

## Goals / Non-Goals

Goals: reproducible graph inventory and real ONNX dialect import first, followed
by a proven IR adapter and compiled CPU semantics. Keep import-only results
distinct from a runnable model package.

Non-goals: vendor compiler, hardware, ISA, numeric-contract changes, automatic
fallback to ORT, or an unverified claim of ARM32 or performance support.

## Decisions

1. Add the integration under `src/export/compilers/onnx_mlir/`. Invoke a pinned
   external compiler rather than copying its source into tracked product code.
2. Start with standalone diagnostics and import APIs, leaving the production
   export selection unchanged until the compatibility gate passes. This makes
   investigation useful without breaking existing packages.
3. Inventory custom domains explicitly; standard-op name support is insufficient
   evidence for a Microsoft-domain fused operator. Record verified conversion
   coverage separately from upstream import capability.
4. Preserve intermediate MLIR and compiler diagnostics in ignored workspaces.
   Reject missing/empty compiler outputs even when a process returns success.
5. Adapt on the export side to the existing partition contract. Imported ONNX
   dialect is not assumed interchangeable with linalg/NPU dialect IR.
6. Keep ONNX-MLIR CPU runtime support separate from the removed ORT session
   dependency. Any remaining runtime library must be explicitly packaged.
7. Preserve fixed CPU rounding at quantization boundaries: ordered division,
   full literal precision, explicit FMA, softmax accumulation lanes and grouped
   decode products. Verify full-width operators and whole-model prefill/decode
   against an independent unchanged-source reference before removing ORT.

## Risks / Trade-offs

- LLVM revision mismatch -> pin source/image revisions and prove textual IR
  compatibility before connecting unchanged vendor tools.
- Fused normalization, rotary embedding, and GQA are not automatic standard-op
  lowerings -> inventory actual domains and validate decompositions independently.
- ARM32 runtime ABI and index widths -> compile and execute a minimal fixture
  before full model integration.
- Large pre-existing staged and unstaged migration -> record the baseline and
  restrict current edits to the selected change.

## Migration Plan

Follow compatibility, operator conversion, export integration, package cleanup,
and board validation in that order. Production ORT remains usable until the
new package passes. Do not automatically clean generated user workspaces.
Any need for vendor edits is a scope blocker requiring a plan revision.
