# Pull Request Rules

Every task branch reaches `dev` through a pull request.

Before opening a pull request:

1. Confirm authentication, remote repository, head branch, and `dev` base.
2. Read `AGENTS.md` and `docs/rules/`.
3. Inspect the entire branch against `dev`, not only the newest commit.
4. Run all applicable lint, simulation, Python, OpenSpec, synthesis, timing,
   and board gates; clearly identify every gate not run.

The pull-request body covers:

- behavior changed and motivation;
- linked OpenSpec change and issues;
- exact validation commands and results;
- numeric-model, register-map, AXI, or format compatibility;
- synthesis, timing, and resource results only when a report exists;
- remaining blockers, risks, and rollback considerations.

Use one PR per issue branch. If review, CI, or validation fails, continue in the
same issue worktree and update the same PR. A closed-but-unmerged PR preserves
the worktree and branch until recovery or abandonment is explicitly chosen.

Task PRs normally target `dev`. GitHub closing keywords only auto-close linked
issues when the PR targets the repository's default branch, so use `Refs
#<issue-id>` for the task PR and verify/close the issue separately after the PR
is merged into `dev`. Only then may the clean worktree and exact local/remote
issue branches be removed under the authorized cleanup rules.

Hard limits:

- Do not push unless publishing the branch or pull request is authorized.
- Never push directly to `main` or `dev`, force-push shared branches, bypass
  hooks, or rewrite published history.
- Never merge a pull request without a separate explicit request.
- Never merge `dev` into `main`; that is a human deployment decision.
- Never claim CI passed until GitHub reports success for the current head SHA.
- Never claim timing closure or board success without objective evidence.
- Never include credentials, licences, private keys, or bitstreams in a pull
  request.
