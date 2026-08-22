---
name: pynq-dev
description: Develop, fix, refactor, and verify PYNQ-Z1 NPU hardware/software changes through an OpenSpec change while enforcing this repository's AGENTS.md and docs/rules contracts. Use for RTL, numeric models, cocotb/testbenches, model export, PYNQ runtime, Vivado Tcl/constraints, overlay integration, MMIO, synthesis preparation, and board validation. Route narrow Vivado installation, file-transfer, or established MAC runbook requests to their specialized skills.
---

# PYNQ Development

Work from the repository root. Treat `AGENTS.md` and applicable files under
`docs/rules/` as hard constraints. Use OpenSpec for every product development
change; do not edit product code from an informal request alone.

## Required context

For every new run or continuation:

1. Reset or resume `STATE.md` according to
   [state-rules.md](references/rules/state-rules.md).
2. Read `AGENTS.md`, `docs/rules/filetree.md`, and
   [env.md](references/rules/env.md).
3. Read [development-matrix.md](references/development-matrix.md) and select
   every affected area before planning validation.
4. Read [validation-gates.md](references/validation-gates.md) before editing or
   claiming completion.
5. Read [filetree.md](references/rules/filetree.md) before adding, moving, or
   deleting files.
6. Read [issue-workflow.md](references/issue-workflow.md) before claiming an
   issue, creating a branch or worktree, committing, opening a PR, or cleaning
   up merged work.

Repository rules win over this skill. Stop and report the exact conflict when a
request cannot satisfy them.

## OpenSpec gate

1. For exploration without implementation, use `openspec-explore`.
2. For a new change, use `openspec-propose`; for an existing change, select it
   explicitly and use `openspec-update-change` when its artifacts are stale.
3. Run OpenSpec status and apply instructions. Read every returned context file.
4. Begin product edits only when the selected change is apply-ready.
5. Use `openspec-apply-change`, implement one task at a time, and mark its
   checkbox complete immediately after objective verification.
6. When all tasks and gates pass, offer `openspec-sync-specs` or
   `openspec-archive-change`; never archive automatically.

If the `openspec` executable is not on PATH, use the repository-approved local
CLI resolution in [env.md](references/rules/env.md). Do not install it globally.

## Issue-scoped development gate

Treat one issue as one development unit. Claim it before implementation, use
the mandatory `npu/issue<issue-id>-<agent-id>` branch from `dev`, and perform all
work in its dedicated worktree. Continue every review or CI fix in that same
worktree and PR. Close and clean up only after the PR is confirmed merged and
the safety checks in [issue-workflow.md](references/issue-workflow.md) pass.

For a large feature, first define one tracking OpenSpec change and parent issue,
then decompose implementation into dependency-linked sub-issues with independent
acceptance evidence. Each implementation sub-issue gets its own normal
issue/worktree/branch/PR lifecycle; do not create one oversized feature branch.

## Development sequence

1. Define scope, non-goals, affected areas, acceptance criteria, and required
   evidence in the OpenSpec artifacts.
2. Freeze affected contracts before implementation: numeric behavior, register
   map, interfaces, file formats, overlay metadata, or board-visible behavior.
3. Establish a failing test or document why test-first evidence is not
   applicable.
4. Make the smallest coherent change inside repository-approved paths.
5. Run every applicable gate from `validation-gates.md`; simulation is not
   evidence of working MMIO or physical-board behavior.
6. Inspect the diff for generated files, secrets, unrelated edits, and contract
   drift. Preserve all user-owned working-tree changes.
7. Update `STATE.md` and the OpenSpec checklist with exact commands, results,
   artifacts, blocked gates, and remaining risks.

## Specialized routing

- Use `build-mac-npu-on-pynq-z1` when work is specifically within its existing
  MAC architecture and end-to-end runbook.
- Use `vivado-design-suite-install` for Vivado installation or licensing.
- Use `send-to-pynq-board` for file transfer to the board.
- Use `ic-design-repo-create` for repository structure or governance changes.

Specialized routing does not remove the OpenSpec gate for product changes and
does not weaken `AGENTS.md` or `docs/rules/`.

## Hard guards

- Never commit directly to `main` or `dev`, merge `dev` into `main`, or create a
  release without the user-requested repository workflow.
- Never commit Vivado projects, bitstreams, build/results directories,
  credentials, private keys, caches, or board-generated files.
- Never change the numeric contract implicitly; RTL, golden model, export, and
  runtime consumers must remain coherent.
- Never claim a blocked Vivado, synthesis, timing, network, or board gate passed.
- Never modify network settings, firewall, routes, board images, tool licenses,
  or running Vivado sessions without explicit authorization.
- Never create duplicate active worktrees for one issue or clean up an issue
  worktree/branch before its PR is confirmed merged.
