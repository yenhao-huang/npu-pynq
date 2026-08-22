## Context

The repository already contains OpenSpec workflow skills and specialized PYNQ
skills, but lacks one entry point that coordinates them for general development.
Repository-local skills use `.codex/skills/` and are classified under `dev/`,
`deploy/`, or `custom/ic_design/` by purpose. See `proposal.md` for motivation and
`specs/pynq-development-workflow/spec.md` for the behavioral contract.

## Goals / Non-Goals

**Goals:**

- Make OpenSpec planning and apply instructions a hard gate before editing.
- Route work to area-specific contracts and proportional validation gates.
- Keep execution resumable and evidence-based through state tracking.
- Remain useful for RTL, Python model/export/runtime, Vivado, and board work.

**Non-Goals:**

- Replace the existing OpenSpec or specialized PYNQ skills.
- Install FPGA tools, alter the board/network, or implement an NPU feature.
- Redesign the repository-wide skill classification beyond the approved
  `dev/`, `deploy/`, and `custom/ic_design/` categories.
- Archive the OpenSpec change automatically; archive remains an explicit
  post-validation action.

## Decisions

### Compose existing workflows instead of duplicating them

`pynq-dev` will act as an orchestrating skill. It will point to the installed
OpenSpec actions and repository commands, and use an affected-area matrix to
select additional references. This keeps detailed OpenSpec mechanics in their
existing skills. The alternative—copying all OpenSpec instructions—would drift
as OpenSpec evolves.

### Use hard phase gates with proportional validation

The skill will gate work as context/rules, OpenSpec readiness, affected-area
contracts, implementation, validation, and handoff. Validation is selected by
impact: RTL requires lint/simulation; numeric changes require golden-model
parity; synthesis and board checks are required only when behavior can reach
those stages. The alternative—requiring a full Vivado/board run for every
documentation or host-only change—would be unnecessarily slow and frequently
blocked.

### Separate concise workflow from detailed references

`SKILL.md` will contain triggers, the core sequence, and hard guards. Detailed
area mapping, verification gates, environment assumptions, and state rules will
live under `references/`. This follows the local skill layout and avoids a
single oversized entry point.

### Follow the repository-local discovery and classification path

The skill will be created at `.codex/skills/custom/ic_design/pynq-dev/`, matching
the repository-local skill discovery layout and the IC-design classification
defined by `docs/rules/filetree.md`.

## Risks / Trade-offs

- [OpenSpec CLI is not installed globally] -> Invoke the locally resolved
  `npx @fission-ai/openspec` command and document CLI availability as an
  environment prerequisite.
- [Vivado/board evidence may be unavailable] -> Preserve partial progress and
  report the exact blocked gate; never substitute simulation for board proof.
- [A broad trigger could overlap specialized skills] -> Route focused install,
  transfer, or established MAC workflows to their specialized skill while
  retaining OpenSpec and repository-rule gates.

## Migration Plan

1. Add the new skill and required state/rules/template files under the approved
   `.codex/skills/custom/ic_design/` category.
2. Validate metadata, required layout, OpenSpec artifacts, and rule references.
3. Leave the change active and ready for explicit archive after review.

Rollback consists of removing only the new `pynq-dev` directory and this
OpenSpec change; no product code or external state is changed.
