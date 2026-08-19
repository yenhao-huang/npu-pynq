# Branch Rules

Three kinds of branch, and one of the two boundaries between them is manual.

```text
main            deploy version; what is built and flashed to the board
  ^  manual merge by a person, never by an agent
dev             integration; everything merges here first
  ^  pull request
<agent>         one branch per agent or per piece of work
```

## main

`main` is the deploy version. The bitstream on the board corresponds to a
commit on `main`, and every release tag points at one.

`main` is protected: no direct pushes, no force-pushes, and the `ci` check must
pass. It only ever advances by a merge from `dev`.

## dev

`dev` is where work integrates. It may be broken briefly; `main` may not.

A merge from `dev` into `main` is a **deploy decision**, so a person makes it by
hand after checking that synthesis, timing, and on-board behaviour are
acceptable. CI cannot verify those on a GitHub-hosted runner, which is why this
step is not automated.

**An agent never merges `dev` into `main`.** An agent may prepare the merge,
report what is ready, and say what it could not verify. The merge itself waits
for a person.

## agent branches

One branch per agent or per piece of work, named for the agent and the task:

```text
claude/saturating-accumulator
claude/export-quantizer
```

Branch from `dev`, and merge back into `dev` through a pull request. Never
branch an agent branch off `main`.

Delete the branch after its pull request merges. A stale agent branch is
indistinguishable from work in progress.

## Releases

Releases are tags on `main`, not a branch: `v0.1.0-<design>`.

A bitstream is immutable and bound to one source revision, one Vivado version,
and one board. "Which commit is on the board?" must have an exact answer, and a
branch tip does not give one. Tag the commit that produced the bitstream and
attach the bitstream to that tag's GitHub Release.

## Rules

- Never commit directly to `main` or `dev`. Work on an agent branch.
- Never force-push any shared branch.
- Rebase an agent branch on `dev` to resolve conflicts; never rebase `dev` or
  `main`.
- Check the current branch before the first commit of a session. If it is
  `main` or `dev`, branch first.
