## Why

Production CD was triggered by a published GitHub Release, so the only way to
learn whether a `main` commit builds, synthesizes and runs on the PYNQ-Z1 was to
publish the Release first. A failed run then left a published, stable Release
that nothing had validated, and v0.1.4 and v0.1.5 show the other half of the
problem: both were promoted to `main` and neither was ever released.

Issue #79 also requires that a user with a PYNQ-Z1 and no development tools can
download one package, extract it, open `resnet18.ipynb`, and classify a real
photograph. The ResNet-18 example was delivered by copying a repository checkout
to the board, and the published Release carried only the matrix example.

## What Changes

- Validate a `release/vMAJOR.MINOR.PATCH` branch cut from `dev`, before
  anything is tagged, published, or merged. A push event uses the workflow from
  the branch it ran on, so a release validates the delivery path as well as
  itself.
- Require the branch name to equal the version `changelog/vMAJOR.MINOR.PATCH.md`
  declares, and refuse a version that is already tagged.
- Order the gates so host checks, the Vivado build, the standalone packages and
  physical PYNQ-Z1 acceptance all pass before any Release object exists.
- Publish a **draft** Release from a passing run, holding the packages, overlay
  artifacts, checksums and board evidence. A draft has no tag and is not public.
- Create the tag only when the draft is published, and only once the validated
  commit is contained in `main`.
- Refuse a promotion into `main` that declares no new release version, so the
  version a candidate may carry is settled in review.
- Assemble a standalone ResNet-18 package that carries the notebook, runtime,
  quantized model, ImageNet labels, demo pictures, overlay and build reports,
  and prove the deployed tree against the published archive digest on the board.
- Add non-interactive real-image acceptance that classifies the pinned
  photograph on the physical overlay and records the input image, the top-5
  prediction and the verdict beside the release identity.

## Capabilities

### New Capabilities

- `main-promotion-validation`: Defines the enforced pre-merge gate for `main`,
  version declaration, and the approval boundaries around promotion.
- `resnet18-release-package`: Defines the standalone ResNet-18 package layout,
  its provenance manifest, and non-interactive real-image board acceptance.

### Modified Capabilities

- `release-continuous-deployment`: Moves validation ahead of every Release
  object by running CD on a release branch, makes a draft Release the output of
  a passing run, separates publication into its own dispatched workflow, and
  extends the published asset set with the ResNet-18 package, checksums, and
  evidence.

## Impact

- `.github/workflows/cd.yml`, `.github/workflows/ci.yml`, the new
  `.github/workflows/release-publish.yml`, and `.github/cd/`.
- A new `release-npu-pynq` skill under `.codex/skills/deploy/` carrying the
  procedure.
- `examples/resnet18/` packaging, board entry points, notebook root detection,
  and tests; `examples/matrix-multiplication/` workflow contract tests.
- `docs/rules/ci-cd.md`, the root README, and the v0.1.5 changelog.
- Repository configuration a person must apply: branch protection on `main`
  with `lint-and-simulate` and `release-readiness` required, a
  `pynq-z1-production` environment for the board runner, and a
  `pynq-z1-release` environment whose required reviewers own the decision to
  make a validated candidate stable.
- No RTL, ABI, or numeric change.
