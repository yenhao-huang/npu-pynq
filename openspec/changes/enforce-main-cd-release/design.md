## Context

See `proposal.md`. `cd.yml` runs on `release: [published]`, validates that the
tag is contained in `main`, and then builds and deploys. Everything it proves
is therefore proved after publication. `docs/rules/ci-cd.md` records that
ordering, and `examples/matrix-multiplication/tests/test_cd_delivery.py`
asserts it.

The ResNet-18 example already has a validated model workspace, a real-image
demo notebook, an acceptance bundle format, and a board runner, but they are
delivered by copying the repository to the board. `examples/resnet18/tests/
test_delivery.py` already describes the standalone package that was never
completed: a reproducible allowlisted archive, digest verification before
execution, and atomic promotion.

## Goals / Non-Goals

**Goals:**

- One validated commit, one immutable tag, one Release, one package, all
  naming the same source revision.
- No stable Release exists until the board has accepted the package.
- A user needs the archive and a board, nothing else.
- Version selection stays a human decision, recorded in the repository.

**Non-Goals:**

- Merging `dev` into `main` automatically. Promotion remains a person's
  decision, as `AGENTS.md` requires.
- Running Vivado or board work for pull requests or for `dev`.
- Committing overlays, model weights, board evidence, or credentials.
- Changing RTL, the hardware ABI, or numeric behaviour.

## Decisions

### Validation comes before any Release object exists

The ordering problem was that validation could only run after something had
already been published. A release branch removes it entirely: `release/vX.Y.Z`
is cut from `dev` and pushed, and CD runs on that branch. Because a push event
uses the workflow file from the branch it ran on, the release validates the
delivery path as well as itself — the first execution of a changed workflow is
never the one that publishes.

A passing run publishes a draft Release. A draft has no tag, is not public, and
can be recreated, so a failed or repeated run costs nothing and leaves nothing
behind. The draft is the artifact of validation: it holds the packages, the
checksums and the board evidence, and it is what the promotion pull request
points at.

Publication is a separate dispatched workflow. It builds nothing, touches no
board, and refuses a draft whose commit is not contained in `main`. The tag is
therefore created last, and it names a commit that is both hardware-validated
and merged.

Alternatives considered. Triggering on a push to `main` and having CD create the
tag: rejected because it takes the publication decision away from the person
making it, and because CD could not be exercised before it was load-bearing.
Triggering on a published prerelease: closer, but the first run of a changed
workflow still had to come from the default branch, so a change to CD could not
be validated before it mattered.

### Version selection is declarative

`.github/cd/resolve_release_version.py` reads `changelog/` and returns the
highest declared `vMAJOR.MINOR.PATCH`. If that tag already exists the run
publishes nothing and succeeds, so ordinary `main` traffic such as a
documentation fix does not fail CD or republish a release. The same script is
the pre-merge gate in `ci.yml`: a promotion into `main` that declares no new
version cannot merge. Version selection therefore happens in review, in a file,
not in a workflow input.

### Human approval sits at three places

`board-validation` uses the protected `pynq-z1-production` environment. Merging
the promotion pull request is a person's decision, as `AGENTS.md` requires.
Publishing the draft is a dispatched workflow in the `pynq-z1-release`
environment, whose required reviewers own the release decision. None of the
three implies another, and everything each decision needs is already attached to
the CD run.

### Squash merges are refused, deliberately

The packages record the commit they were built from, and the tag is created at
that commit. A squash merge produces a different commit, leaving the validated
one absent from `main`, so publication checks containment and fails with an
explanation rather than tagging something unvalidated.

### Failure recovery

Nothing is tagged or public until the last step, so recovery is ordinary work:
fix the cause on `dev`, rebuild the release branch, push again, and the run
recreates the draft. Publishing a draft by hand to finish a run defeats the
containment check, so the rules say not to; tags are never moved or deleted.

### The package is the deployment unit

The board receives exactly the archive that is published, and
`verify_package_tree` proves the extracted tree against that archive's SHA-256
and against the per-file digests in `package.manifest.json` before the overlay
is programmed. An unexpected file fails the run. Generated evidence files are
the only additions the board may make.

Layout is flat and self-describing: `resnet18.ipynb`, `model/`, `artifacts/`,
`reports/`, `acceptance/`, and `src/` under one root with
`package.manifest.json`. The notebook finds its root by that manifest, so the
same notebook works from the archive and from an ordinary deployment.

### Real-image acceptance is separate from the demonstration

The notebook stays a demonstration. `accept_image_on_board.py` is the
non-interactive counterpart: it reloads the host-prepared INT8 tensor, runs it
on the physical overlay, compares every capture against the host record byte
for byte, decodes the top-5, and fails when the declared class does not win.
Its evidence names the release tag, the source commit, the overlay digests, the
image and its provenance, and the prediction, which is what Issue #79 requires
board evidence to identify.

## Risks / Trade-offs

- Pushing a release branch starts a multi-hour privileged run. The concurrency
  group serializes runs, and the branch name must match a declared version, so a
  run cannot start by accident.
- The release branch is a fourth long-lived-looking ref during a release. It is
  short-lived: it exists between the cut and the promotion merge, and carries the
  same tree as the `dev` commit it was cut from.
- The self-hosted runners must hold the untracked ResNet-18 model workspace and
  acceptance bundle. The packaging step fails closed when they are absent, so
  the failure is a missing input, never a silently smaller package.
- Creating the draft requires `contents: write` on one CD job, and publishing
  requires it on one job of the publication workflow. Both are scoped to a single
  job, and publication is additionally gated by an environment with required
  reviewers.
