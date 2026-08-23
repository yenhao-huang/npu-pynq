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

## Merge strategy

Preserve logical changes and squash development iterations. **Squash and
merge** is the default for most single-issue task PRs so that one issue is
normally represented by one Conventional Commit on `dev`. Fold iterative
`docs`, `chore`, `test`, and `ci` commits into the logical change they complete,
including review or CI follow-ups that have no independent value.

Do not choose the strategy from commit type alone. Preserve multiple commits
when the PR contains independently reviewable, revertible, or bisectable
logical changes, such as a substantive feature, bug fix, performance change,
refactor, build-system change, or new regression capability. Keep `revert`
commits separate. The PR must make the intended mainline commit subject and
any intentionally preserved commit boundaries clear to reviewers.

Applying this policy does not authorize history rewriting or merging. Clean up
only private, unpublished history when safe; for a published task branch, use
GitHub's approved merge strategy without force-pushing. The hard limits below
still require a separate explicit request before any pull request is merged.

Task PRs normally target `dev`. GitHub closing keywords only auto-close linked
issues when the PR targets the repository's default branch, so use `Refs
#<issue-id>` for the task PR and verify/close the issue separately after the PR
is merged into `dev`. Only then may the clean worktree and exact local/remote
issue branches be removed under the authorized cleanup rules.

## Release after merging into `main`

Every approved `dev` → `main` deployment merge MUST be followed by a GitHub
Release. The release tag MUST point to the merge result on `main`, never to a
feature branch or `dev`.

Use semantic version tags in the form `vMAJOR.MINOR.PATCH`. The default patch
release sequence is:

```text
v0.1.0 → v0.1.1 → v0.1.2 → ...
```

Unless a release decision explicitly selects a new major or minor version,
increment only `PATCH` from the latest existing tag. Before tagging, fetch the
latest `main` and verify that the working tree is clean:

```bash
git fetch origin --tags
git switch main
git pull --ff-only origin main
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

Replace `v0.1.0` with the next unused version. Create the GitHub Release from
that tag and attach deployable BIT/HWH/provenance artifacts when applicable.
Vivado projects, generated reports, credentials, and board secrets MUST NOT
be committed to Git. Record the exact tag, commit, validation evidence, and
known blockers in the release notes.

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
