# Branch Rules

```text
main            deploy version; what may be built and flashed to the board
  ^  manual merge by a person after deploy validation
dev             integration branch
  ^  pull request
<task branch>   one branch per agent, contributor, or logical change
```

## Main

`main` is protected: no direct pushes, no force-pushes, and required checks must
pass. Releases and deployable board revisions are tags on `main`.

## Dev

All normal changes integrate into `dev` first. A merge from `dev` to `main` is
a deployment decision made by a person after synthesis, timing, and board
behavior have been reviewed. Agents may prepare and report this merge but must
never perform it.

## Task branches

Create one branch from `dev` for each claimed issue. Its mandatory name is:

```text
npu/npu-<issue-id>-<agent-id>
```

For example, issue 142 claimed by `codex-a` uses
`npu/npu-142-codex-a`. Keep `agent-id` stable, lowercase, and kebab-case for the
issue. Create or attach one dedicated worktree outside the primary checkout,
perform all issue development and PR fixes there, and never create a duplicate
active worktree for the same issue.

Merge task branches into `dev` through pull requests. Delete the worktree and
branch only after GitHub confirms the PR merged and cleanup safety checks pass.
Never branch task work from `main`.

## Releases

Use immutable tags on `main`, such as `v0.1.0-<design>`. Each released
bitstream must identify its source tag, Vivado version, target board, and
matching hardware handoff.

## Hard rules

- Never commit directly to `main` or `dev`.
- Never force-push a shared branch.
- Rebase a private task branch on `dev` when needed; never rebase `dev` or
  `main`.
- Check the current branch before the first commit of every session.
- Never remove an issue worktree or delete its branch while its PR is open,
  unmerged, or still required for review/CI fixes.
- Never merge `dev` into `main` without an explicit human deployment decision.
