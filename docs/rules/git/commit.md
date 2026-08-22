# Commit Rules

Use Conventional Commits based on the actual diff:

```text
<type>(<scope>): <imperative description>

<why and validation context>

<issue footer>
```

Types include `feat`, `fix`, `perf`, `test`, `refactor`, `docs`, `build`, `ci`,
`chore`, and `revert`. Use a design name as scope for focused hardware work;
otherwise use `hw`, `test`, `export`, `runtime`, `examples`, `docs`, or `ci`.

The subject is imperative, present tense, has no trailing period, and stays
under 72 characters. The body explains why, numeric or interface consequences,
measured timing/resource impact when available, and which tests cover the
change.

Mark changes to the register map, AXI interface, serialized model format, or
numeric contract as breaking:

```text
feat(conv)!: widen accumulator to 48 bits

BREAKING CHANGE: saturation bounds changed; re-export existing models.
```

Rules:

- One logical change per commit.
- Preserve logical changes and squash development iterations. Commits that are
  independently reviewable, revertible, or useful for `git bisect` remain
  separate; follow-up commits that only complete or correct the same unmerged
  change should become one mainline commit.
- Within one issue or pull request, aggressively squash iterative `docs`,
  `chore`, `test`, and `ci` commits when they describe one logical change.
  Do not squash solely because of the commit type: a new regression suite or
  CI capability can be an independent logical change.
- Keep substantive `feat`, `fix`, and `perf` changes separate when each has its
  own historical value. Trivial `refactor` and `build` cleanup may be squashed;
  substantive refactors or build-system changes should remain separate.
- A fix or test added while an unmerged feature is still being developed is
  normally part of that feature. For example, an arbiter implementation,
  reset correction, synthesis-warning cleanup, and its tests may become one
  `feat(...): add arbiter` commit.
- Keep every `revert` as a separate commit because it records an explicit
  historical event.
- Do not commit generated artifacts, credentials, licences, or board secrets.
- Reference task issues with `Refs #123`. Task branches merge into `dev`, so
  verify and close the issue separately after the PR merges and acceptance
  criteria pass; do not rely on a default-branch closing keyword.
- RTL changes without a test change must explain why coverage is unchanged.
- Never bypass hooks with `--no-verify`.
- Never rewrite published history or run destructive Git commands without an
  explicit request naming the operation.

Clean up private, unpublished commit history before publication when safe. If
the branch has already been published, do not rebase or force-push it to apply
these rules; use the pull request's approved squash merge strategy instead.
