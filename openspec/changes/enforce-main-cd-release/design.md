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

### The stable Release is an output; the prerelease is the trigger

Publication still comes first, which keeps the familiar order: a person decides
a candidate is ready and publishes it. What changes is what they publish. The
candidate goes out as a prerelease, CD validates it, and the last action of a
passing run clears the prerelease flag. A stable Release therefore always means
the board accepted the packages attached to it.

CD refuses a Release that is already stable, because validating one would attach
evidence to something that was already public and unvalidated, which is the
failure this change exists to remove.

Converting a prerelease to a stable Release raises GitHub's `released` event,
not `published`, so the promotion at the end of a run cannot retrigger CD. The
workflow listens only to `published`.

Alternative considered: trigger on a push to `main` and have CD create the tag
and Release itself. Rejected because it takes the publication decision away from
the person making it, and because the ordering people already have in their
heads — publish, then watch it validate — is worth keeping when it can be made
safe.

Only `promote-release` holds `contents: write`, so no earlier job can modify the
Release or the repository even if it is compromised.

### Version selection is declarative

`.github/cd/resolve_release_version.py` reads `changelog/` and returns the
highest declared `vMAJOR.MINOR.PATCH`. If that tag already exists the run
publishes nothing and succeeds, so ordinary `main` traffic such as a
documentation fix does not fail CD or republish a release. The same script is
the pre-merge gate in `ci.yml`: a promotion into `main` that declares no new
version cannot merge. Version selection therefore happens in review, in a file,
not in a workflow input.

### Human approval sits at promotion

`board-validation` uses the protected `pynq-z1-production` environment, and
`promote-release` uses `pynq-z1-release`. Configuring required reviewers on the
latter keeps the decision to make a candidate stable with a person, while
everything that decision needs — build reports, board evidence, the real-image
result — is already attached to the run.

### Failure recovery

A failed run leaves the candidate as a prerelease. That is a visible, honest
state: it says a candidate existed and was not accepted. Fix the cause, promote
again, and publish a prerelease for the next version. Clearing a prerelease flag
by hand to "finish" a run defeats the whole arrangement, so the rules say not to;
tags are never moved or deleted.

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

- Publishing a prerelease starts a multi-hour privileged run. The concurrency
  group serializes runs, and the pre-merge gate settles the version before the
  candidate can be published at all.
- A prerelease is publicly visible while it is being validated. That is the cost
  of keeping publication first; it is labelled as a prerelease throughout, and it
  carries no assets until the run attaches them.
- The self-hosted runners must hold the untracked ResNet-18 model workspace and
  acceptance bundle. The packaging step fails closed when they are absent, so
  the failure is a missing input, never a silently smaller package.
- Editing the Release from CI requires `contents: write` on one job. It is
  scoped to `promote-release` and gated by an environment with required
  reviewers.
