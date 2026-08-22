## Context

The current roadmap records development governance but says no product
milestones are confirmed. The user has now confirmed eight phase-level NPU
milestones and explicitly requested that the roadmap contain only this broad
direction. `docs/rules/human-docs.md` requires exact-path, exact-batch approval;
that approval applies only to `docs/human/roadmap.md` in this change.

## Goals / Non-Goals

**Goals:**

- Preserve the existing confirmed development direction.
- Replace the empty product-milestone statement with the eight confirmed phases.
- Keep each phase to a concise title and outcome.
- Present planned product milestones before development-governance direction.

**Non-Goals:**

- Detailed architecture, numerical formats, acceptance tests, task breakdowns,
  dates, owners, dependencies, or performance commitments.
- Changes to feature inventory, changelog, README, product code, or board state.

## Decisions

- Use one bullet per phase under `Planned milestones`. This is easier to scan
  and matches the user's request for high-level direction; detailed nested
  checklists were rejected as too granular for a roadmap.
- Order `Planned milestones` before `Confirmed direction` because product phases
  are the roadmap's primary information. The alternative governance-first order
  makes readers pass operating rules before reaching the plan.
- Preserve the user's phase identifiers (`0`, `1A`, `1B`, `1C`, `2A`, `2B`,
  `2C`, `3`) verbatim so later OpenSpec changes and issues can reference them.
- Keep technical identifiers such as `examples/matrix_multiplication`, ResNet-18,
  and Small Transformer because they define milestone boundaries without
  prescribing implementation.

## Risks / Trade-offs

- [Phase descriptions are intentionally broad] -> Put detailed requirements,
  architecture, dependencies, and acceptance evidence in future OpenSpec
  changes and issues rather than expanding the roadmap.
- [No schedule is shown] -> Add dates only after separate human confirmation;
  this update creates no delivery commitment.

## Migration Plan

Replace only the placeholder sentence under `Planned milestones`, then move the
complete milestone section before `Confirmed direction`. Read the file back,
validate heading and phase order, and leave all other human documents unchanged.
Rollback restores the original section order and placeholder sentence.
