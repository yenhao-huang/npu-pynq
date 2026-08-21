# Issue-Scoped Development Workflow

Use one GitHub issue as the ownership, scope, branch, worktree, pull-request,
and cleanup unit. Do not combine unrelated issues in one branch or split one
issue across multiple active worktrees unless the issue explicitly defines
coordinated sub-issues.

## Large-feature decomposition

Before implementation of a large feature:

1. Create or select one tracking OpenSpec change that defines the complete
   behavior, non-goals, architecture boundaries, and end-to-end acceptance.
2. Use one parent tracking issue to link the OpenSpec change, dependency graph,
   integration criteria, and implementation sub-issues.
3. Split work into sub-issues that can be owned, implemented, tested, reviewed,
   and merged independently. Record `Blocked by`, `Related to`, or sub-issue
   relationships instead of relying on execution order in prose.
4. Give each implementation sub-issue its own worktree, branch, and PR through
   the workflow below. A tracking-only parent issue does not need an
   implementation branch unless it owns separate integration work.
5. Close the parent only after all required sub-issues merge and the parent
   OpenSpec/end-to-end acceptance gates pass.

Do not use one long-lived feature branch as a substitute for decomposition.

```text
Issue created
  -> agent claims issue
  -> create issue worktree and branch
  -> develop through the linked OpenSpec change
  -> commit and test
  -> open PR to dev
  -> wait for review, CI, and authorized merge
       PR not merged -> continue in the same worktree
       PR merged     -> verify/close issue -> remove worktree -> delete branch
```

## 1. Select and claim the issue

1. Confirm the exact repository, GitHub authentication, issue number, issue
   state, acceptance criteria, dependencies, assignees, and linked OpenSpec
   change.
2. Search for an existing branch, worktree, or open PR for the issue. Reuse the
   same active worktree instead of creating a duplicate.
3. Claim through the repository's supported mechanism: assign the authenticated
   GitHub user when appropriate and add a concise comment containing the stable
   `agent-id`. Do not invent labels or remove other assignees.
4. Read the issue back after claiming. If another owner has already claimed it,
   or ownership is ambiguous, stop and resolve ownership before branching.

`agent-id` is lowercase kebab-case and stable for the issue. Record the issue
number, URL, owner, claim evidence, and agent id in `STATE.md`. Claiming an issue
does not authorize changing its scope, closing it, pushing, opening a PR, or
merging.

## 2. Create the branch and worktree

The branch name is mandatory:

```text
npu/npu-<issue-id>-<agent-id>
```

Example: `npu/npu-142-codex-a`.

Create it from `dev`, never from `main`. Use one atomic `git worktree add -b`
operation when the branch does not exist. Put the worktree outside the primary
checkout and choose a deterministic path containing the issue id and agent id.
Before creation, verify that the exact branch and path are unused and that the
selected `dev` revision is the intended base.

If the branch already exists, validate its issue identity and attach a worktree
to that branch; never silently reset, recreate, or overwrite it. Perform all
issue development, OpenSpec updates, tests, commits, and PR fixes inside this
worktree. Do not edit the primary checkout for issue implementation.

## 3. Develop and validate

1. Use the issue's linked OpenSpec change. Create one when the behavior change
   requires it and none exists.
2. Keep issue acceptance criteria, OpenSpec tasks, commits, and tests aligned.
3. Commit one logical change at a time with Conventional Commits and `Refs
   #<issue-id>`.
4. Run all impact-based gates from `validation-gates.md`; record exact commands,
   results, skipped gates, and blockers.
5. Before publishing, verify the current worktree, branch name, expected diff,
   upstream state, and complete branch diff against `dev`.

## 4. Open and maintain the pull request

Open one PR from the issue branch to `dev` only when push and PR creation are
authorized. Link the issue and OpenSpec change, include full validation evidence,
and use `Refs #<issue-id>` because GitHub closing keywords only auto-close an
issue when the PR targets the repository's default branch. This repository's
normal task PR targets `dev`, so issue closure is a separate verified step.

After opening the PR:

- Wait for review and CI for the current head SHA.
- Never merge without a separate explicit request.
- If review, CI, or validation fails, return to the same worktree and branch,
  fix, test, commit, and update the same PR.
- Do not create a replacement worktree or branch merely because review started.
- A closed-but-unmerged PR is not completion; preserve its worktree and branch
  until the user chooses recovery or abandonment.

## 5. Complete and clean up

Cleanup begins only after GitHub reports the PR merged into `dev` and the issue
acceptance criteria are satisfied.

1. Read back the PR state, merged head/base, current CI state, and issue state.
2. If the issue is still open, close it with a link to the merged PR only when
   the current request explicitly authorizes lifecycle completion. Otherwise
   report that manual closure remains.
3. Verify the issue worktree has no uncommitted changes, untracked valuables,
   unpushed commits, or other branches checked out.
4. Resolve and verify the exact worktree path is the issue worktree, then remove
   that worktree. Never use a broad or computed recursive deletion.
5. Delete only the exact local issue branch after the worktree is removed and
   merged state is proven. If safe deletion refuses because the PR used squash
   or rebase merge, do not force-delete without explicit authorization.
6. Verify whether the remote branch was already deleted. Delete the exact remote
   issue branch only when remote cleanup is authorized and no open PR depends on
   it.
7. Record issue, PR, merge SHA, removed worktree path, and local/remote branch
   cleanup outcomes in `STATE.md`.

Never close the issue, remove the worktree, or delete either branch while the PR
is open, closed without merge, failing required checks, or awaiting an
authorized merge.
