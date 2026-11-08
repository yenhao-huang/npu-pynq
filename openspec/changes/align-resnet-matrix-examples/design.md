# Design

## Context

See proposal.md for motivation. Existing package hashes, pinned download
validation, matrix tiling and image predictions must survive relocation.
Source must never import example Python or runtime import export.

## Goals / Non-Goals

Goals: source-owned entry points, standalone board packages, concise paired
guides, unchanged notebook demonstrations and discovery of relocated tests.
Non-goals: arithmetic, RTL, ISA, release publication or changing model weights.

## Decisions

- Put matrix tiling in runtime/core/matrix.py and physical acceptance in
  test/matrix_acceptance.py. Keep vendor DMA unchanged.
- Put image execution in runtime/image_cli.py; share delivery hash checking in
  runtime/core/delivery.py so board execution does not import a host exporter.
- Put builders in export/delivery/{matrix,resnet18}.py. They read notebooks and
  guides as data and assemble source-owned runtime modules into standalone payloads.
- Move pinned metadata and download modules into export/graphs/resnet18; replace
  the conversion wrapper with the existing export CLI. Default products to dist.
- Move automated PowerShell delivery into .github/cd and regression tests into
  src/test/tests. Update all active callers in the same change.

## Risks / Trade-offs

Import roots and package layouts may drift: test isolated archives with no
PYTHONPATH and inspect CD commands. Preserve ignored old model products rather
than deleting them. Notebook paths must work from both example and package roots.

## Migration Plan

Relocate with callers and tests together, then rewrite guides. Run focused
delivery/download/runtime tests, notebook syntax/output checks and source boundary
tests. Use physical board smoke checks when usable existing artifacts are present;
record exactly what was run. Rollback is reverting source changes; no board image,
network or credentials are modified.
