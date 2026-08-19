# Pull Request Rules

Every agent branch reaches `dev` through a pull request. See
[git-branch.md](git-branch.md) for what may merge where.

## Hard limits

- Do not push unless the user asked for a PR or a push.
- Never push to `main` or `dev` directly, and never force-push either.
- Never merge a PR unless the user separately and explicitly asks.
- Never merge `dev` into `main`; that is a person's decision.
- Never rewrite published history or bypass hooks.

## Before opening

1. `gh auth status`, and confirm the base branch is `dev`.
2. Read `AGENTS.md` and `docs/rules/` for repository conventions.
3. Inspect the whole branch, not the last commit:

```bash
git status --short --branch
git log --oneline --decorate --no-merges dev..HEAD
git diff --stat dev...HEAD
git diff dev...HEAD
```

4. Run the repository's validation and record the exact commands and output:

```bash
make -C src/test lint
make -C src/test sim
```

## Creating

```bash
git push -u origin HEAD

gh pr create \
  --base dev \
  --title "PR title" \
  --body-file /path/to/pr-body.md

gh pr view --json number,title,url,state,isDraft,baseRefName,headRefName,statusCheckRollup
```

Use `gh pr edit` for an existing PR and change only the requested fields.

## Body

Describe the whole diff, not the newest commit. Cover:

- what behaviour changed, and why
- which tests ran, with their result; label anything not run and say why
- for RTL changes, whether `src/test/model/` changed with it, and whether the
  register map, AXI interface, or numeric contract moved
- synthesis, timing, or resource results **only if a synthesis run produced
  them**; otherwise state that no synthesis was run
- issue linkage: `Closes #123` only when merging resolves it, `Refs #123`
  otherwise

## Rules

- Never claim CI passed until GitHub reports success for the current head SHA.
- Never claim timing closure or resource numbers without a Vivado report.
  GitHub-hosted runners cannot run Vivado, so CI never produces them.
- Never include credentials or a bitstream in a PR.
- Report the PR URL, number, base and head branches, check state, and what
  still blocks review.
