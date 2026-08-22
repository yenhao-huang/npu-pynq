## Why

PYNQ-Z1 development currently spans RTL, Python models, export/runtime code,
Vivado, simulation, and physical-board verification, but there is no single
repo-local skill that consistently routes changes through OpenSpec and enforces
the repository rules. A dedicated `pynq-dev` skill will make that development
path repeatable and evidence-driven.

## What Changes

- Add a repo-local `pynq-dev` skill for PYNQ/NPU feature work, fixes,
  refactoring, verification, synthesis preparation, and board integration.
- Require every implementation to use an OpenSpec change from proposal through
  apply, with archive offered only after all required validation succeeds.
- Require `AGENTS.md`, `docs/rules/filetree.md`, and relevant skill references
  to be read before edits, with repository rules taking precedence.
- Add development gates for scope selection, numeric/RTL contracts, tests,
  generated-artifact hygiene, board evidence, and handoff reporting.
- Add state tracking and a reusable state template so incomplete or blocked
  work cannot be reported as complete.

## Capabilities

### New Capabilities

- `pynq-development-workflow`: Defines the required OpenSpec-driven,
  repository-compliant workflow for developing and validating PYNQ-Z1/NPU
  changes.

### Modified Capabilities

None.

## Impact

- Adds files under `.codex/skills/custom/ic_design/pynq-dev/` and this OpenSpec change.
- Reuses the existing OpenSpec skills and the repository's simulation, Vivado,
  export, runtime, and board-deployment conventions.
- Introduces no runtime dependency and does not modify RTL, Python product code,
  CI, generated Vivado outputs, credentials, or board state.
