# Roadmap

> Human-maintained document. Any change requires explicit human confirmation.

## Planned milestones

- **Phase 0 — Contracts and feasibility:** Define the numerical contract,
  performance model, and hardware ABI.
- **Phase 1A — Compute core:** Implement processing elements and the systolic
  array with bit-accurate verification.
- **Phase 1B — Board vertical slice:** Integrate DMA, the control interface,
  Tcl generation, bitstream creation, and an on-board vertical slice.
- **Phase 1C — Matrix multiplication:** Deliver
  `examples/matrix_multiplication` through the Python runtime.
- **Phase 2A — ResNet enablement:** Implement ResNet operators, tiling, memory
  planning, and the exporter.
- **Phase 2B — ResNet-18 acceptance:** Complete ResNet-18 accuracy,
  performance, and on-board acceptance.
- **Phase 2C — Production hardening:** Harden the complete NPU hardware and
  software stack for production-level operation.
- **Phase 3 — Small Transformer:** Run a Small Transformer feasibility spike.

## Confirmed direction

- Use issue-scoped development with one worktree, branch, and PR per
  implementation issue.
- Use Spec-Driven Development so behavior and acceptance criteria exist before
  implementation.
- Decompose large features into a tracking OpenSpec change, a parent issue, and
  independently verifiable dependency-linked sub-issues.
- Keep features, roadmap decisions, and weekly change history under explicit
  human control.
