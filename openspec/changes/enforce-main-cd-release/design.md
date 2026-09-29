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
- No tag or Release exists until the board has accepted the package.
- A user needs the archive and a board, nothing else.
- Version selection stays a human decision, recorded in the repository.

**Non-Goals:**

- Merging `dev` into `main` automatically. Promotion remains a person's
  decision, as `AGENTS.md` requires.
- Running Vivado or board work for pull requests or for `dev`.
- Committing overlays, model weights, board evidence, or credentials.
- Changing RTL, the hardware ABI, or numeric behaviour.

## Decisions

### The Release is an output, not a trigger

`cd.yml` runs on `push` to `main`. `publish-release` depends on
`select-version`, `host-checks`, `build-overlay`, and `board-validation`, so a
failure at any gate leaves no tag and no Release. Only `publish-release` holds
`contents: write`, so no earlier job can write to the repository even if it is
compromised.

Alternative considered: keep the release trigger and add a separate validation
workflow on `main`. Rejected because nothing then prevents a person from
publishing a Release the validation workflow never approved.

### Version selection is declarative

`.github/cd/resolve_release_version.py` reads `changelog/` and returns the
highest declared `vMAJOR.MINOR.PATCH`. If that tag already exists the run
publishes nothing and succeeds, so ordinary `main` traffic such as a
documentation fix does not fail CD or republish a release. The same script is
the pre-merge gate in `ci.yml`: a promotion into `main` that declares no new
version cannot merge. Version selection therefore happens in review, in a file,
not in a workflow input.

### Human approval sits at publication

`board-validation` uses the protected `pynq-z1-production` environment, and
`publish-release` uses `pynq-z1-release`. Configuring required reviewers on the
latter keeps the final publication a person's decision, while everything the
decision needs — build reports, board evidence, the real-image result — is
already attached to the run.

### Failure recovery

A failed run publishes nothing; the next push to `main` re-runs every gate. A
run that failed after tagging cannot exist, because tagging is the last step
before publication and `publish-release` refuses a tag that already exists. To
redo a published release, a new patch version is declared with a new changelog
file; tags are never moved or deleted.

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

- A push to `main` that declares a new version starts a multi-hour privileged
  run. The concurrency group serializes runs, and the pre-merge gate makes the
  version explicit before the push happens.
- The self-hosted runners must hold the untracked ResNet-18 model workspace and
  acceptance bundle. The packaging step fails closed when they are absent, so
  the failure is a missing input, never a silently smaller package.
- Tagging from CI requires `contents: write` on one job. It is scoped to
  `publish-release` and gated by an environment with required reviewers.
