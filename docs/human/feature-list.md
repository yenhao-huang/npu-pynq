# Feature List

> Human-maintained document. Any change requires explicit human confirmation.

## `pynq-dev`

### Issue-scoped development

Each implementation issue has one owner, one active worktree, branch
`npu/npu-<issue-id>-<agent-id>`, and one pull request to `dev`. Review and CI
fixes continue in the same worktree; cleanup begins only after merge is proven.

### Spec-Driven Development

Behavior and acceptance criteria are defined before product code. Major work
uses an OpenSpec proposal, requirements, design decisions, and implementation
tasks as the contract for development and verification.

### Large-feature decomposition

A large feature starts with one tracking OpenSpec change and a parent tracking
issue. It is decomposed into dependency-linked sub-issues small enough to
implement and verify independently. Each implementation sub-issue follows the
normal issue/worktree/branch/PR lifecycle; the parent closes only after the
required sub-issues and end-to-end acceptance criteria complete.

## `ic-design-repo-create`

### Reproducible IC repository structure

Separates human-authored RTL, testbenches, constraints, Tcl, software, and
documentation from generated Vivado projects and build products.

### Verification and delivery governance

Provides simulation, CI/CD, Git branch, issue, pull-request, generated-artifact,
and release boundaries suitable for FPGA development.

### Human-owned project communication

Creates `AGENTS.md` plus `docs/human/feature-list.md`, `roadmap.md`, and weekly
changelogs. Agents can read these documents freely, but any mutation requires
explicit human confirmation for the exact proposed batch.
