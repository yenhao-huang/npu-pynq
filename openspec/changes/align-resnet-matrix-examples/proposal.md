# Proposal

## Why

ResNet18 and matrix multiplication still own runtime, download, delivery and test
code inside examples. Their guides cannot follow the current create-example rule
until those reusable responsibilities move into source and CD directories.

## What Changes

- Relocate existing helpers into export, runtime, test and CD owners.
- Update standalone package contents, automated callers and notebook imports.
- Keep only README.md, README.macos.md and an output-free notebook per example.
- Stage generated model assets in dist/resnet18/model.

## Capabilities

No new capabilities or numeric behavior: this is a source organization and
documentation refactor. Specifications are intentionally skipped.

## Impact

Examples, host package builders, asset download entry points, runtime image CLI,
matrix acceptance, test discovery and CD. Hardware, ISA and compiled package
formats remain unchanged. Existing helper file paths are replaced by src modules.
