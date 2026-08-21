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
- Do not commit generated artifacts, credentials, licences, or board secrets.
- Reference task issues with `Refs #123`. Task branches merge into `dev`, so
  verify and close the issue separately after the PR merges and acceptance
  criteria pass; do not rely on a default-branch closing keyword.
- RTL changes without a test change must explain why coverage is unchanged.
- Never bypass hooks with `--no-verify`.
- Never rewrite published history or run destructive Git commands without an
  explicit request naming the operation.
