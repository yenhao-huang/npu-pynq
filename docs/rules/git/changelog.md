# Changelog Rules

Top-level `changelog/` files provide commit-bounded release traceability. They
are contributor-maintained release records and are separate from the
human-owned weekly notes under `docs/human/changelog/`.

## Required files and names

- `changelog/vMAJOR.MINOR.PATCH.md` records the exact change associated with
  that released or proposed version.
- The current upload uses its selected target version directly; do not create
  an `unreleased.md` alias.
- Preserve an earlier version file as the baseline and create a new version
  file for each later upload.

Every file uses these sections in this order:

```text
Change commits
Roadmap progress reached
Issues and corresponding pull requests
```

## Change commits

Record both boundaries as full 40-character commit SHAs:

- **Change-before commit** identifies the repository state before the recorded
  change.
- **Change-after commit** identifies the last implementation or integration
  commit included in the recorded change.

A whole-history baseline uses the repository root as its change-before commit,
marks that boundary as inclusive, and uses the current `main` commit as its
change-after commit. A changelog cannot contain its own commit SHA, so a
changelog-only commit after the integration boundary is not used as the
change-after value.

## Roadmap progress reached

List only progress supported by merged code, accepted evidence, or an explicit
human decision. State incomplete, deferred, closed-without-implementation, and
provenance-limited work plainly. Link the evidence through issue and PR
numbers; do not infer completion from a branch name or an open PR.

## Issues and corresponding pull requests

Use a separate Markdown table for each type batch. Classify a row from the
pull request's Conventional Commit title, keep rows of the same type together,
and use this priority order:

```text
feat, fix, perf, refactor, docs, test, build, ci, chore, revert, release, other
```

The batching rules are:

1. Put `feat` and `fix` before every other category.
2. A batch contains at most five body rows.
3. Emit up to five `feat` rows, then up to five `fix` rows. If either type has
   more rows, repeat `feat` then `fix` batches before lower-priority types.
4. Keep the remaining types in the priority order above, also using at most
   five rows per batch.
5. Within a batch, sort by roadmap order and then issue number, not by the time
   an agent happened to open the PR.

Each row contains the Issue, Pull request, Relationship, and Result columns.
Use `Direct`, `Follow-up`, `Integrated dependency`, `Promotion`, or a similarly
precise relationship. Use an em dash and state `No linked issue` or `No merged
implementation PR` when a one-to-one mapping does not exist. Never invent a
link merely to fill a cell.

Before committing, verify both SHAs exist, every URL resolves to the intended
repository object, no table batch exceeds five rows, and `git diff --check`
passes.
