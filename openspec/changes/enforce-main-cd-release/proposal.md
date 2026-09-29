## Why

Production CD is triggered by a published GitHub Release, so the only way to
learn whether a `main` commit builds, synthesizes, and runs on the PYNQ-Z1 is
to publish the Release first. That is a cycle: the evidence a release decision
needs can only be produced after that decision has already been announced. A
failed run then leaves a published Release that nothing has validated, and
v0.1.4 shows the other half of the problem, where a promotion to `main` was
never released at all.

Issue #79 also requires that a user with a PYNQ-Z1 and no development tools can
download one package, extract it, open `resnet18.ipynb`, and classify a real
photograph. Today the ResNet-18 example is delivered by copying a repository
checkout to the board, and the published Release carries only the matrix
example.

## What Changes

- Trigger production CD on every push to `main` instead of on a published
  Release, and make the Release the last step of a passing run.
- Select the release version from the `changelog/vMAJOR.MINOR.PATCH.md` the
  promotion declares, refuse an already-tagged version, and refuse a promotion
  that declares none before it can merge into `main`.
- Order the gates so that host checks, the Vivado build, the standalone
  packages, and physical PYNQ-Z1 acceptance all pass before any tag exists.
- Create the tag and publish the Release from the validated commit, with the
  packages, overlay artifacts, checksums, and board evidence attached.
- Assemble a standalone ResNet-18 package that carries the notebook, runtime,
  quantized model, ImageNet labels, demo pictures, overlay, and build reports,
  and prove the deployed tree against the published archive digest on the board.
- Add non-interactive real-image acceptance that classifies the pinned
  photograph on the physical overlay and records the input image, the top-5
  prediction, and the verdict beside the release identity.

## Capabilities

### New Capabilities

- `main-promotion-validation`: Defines the enforced pre-merge gate for `main`,
  version declaration, and the approval boundaries around promotion.
- `resnet18-release-package`: Defines the standalone ResNet-18 package layout,
  its provenance manifest, and non-interactive real-image board acceptance.

### Modified Capabilities

- `release-continuous-deployment`: Replaces release-triggered CD with
  main-triggered validation that publishes the Release itself, and extends the
  published asset set with the ResNet-18 package, checksums, and evidence.

## Impact

- `.github/workflows/cd.yml`, `.github/workflows/ci.yml`, and `.github/cd/`.
- `examples/resnet18/` packaging, board entry points, notebook root detection,
  and tests; `examples/matrix-multiplication/` workflow contract tests.
- `docs/rules/ci-cd.md`, the root README, and the v0.1.5 changelog.
- Repository configuration a person must apply: branch protection on `main`
  with `lint-and-simulate` and `release-readiness` required, a
  `pynq-z1-production` environment for the board runner, and a
  `pynq-z1-release` environment whose required reviewers own the publication
  decision.
- No RTL, ABI, or numeric change.
