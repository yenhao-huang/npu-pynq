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

- Keep publication as the trigger, but make the candidate a **prerelease**. CD
  validates it and promotes it to a stable Release; a stable Release is never
  published by hand.
- Refuse a draft, refuse an already-stable Release, and refuse a tag whose
  commit is not contained in `main` or that disagrees with the version
  `changelog/vMAJOR.MINOR.PATCH.md` declares, all before any privileged work.
- Order the gates so host checks, the Vivado build, the standalone packages and
  physical PYNQ-Z1 acceptance all pass before the prerelease flag is cleared.
- Attach the packages, overlay artifacts, checksums and board evidence to the
  candidate, then clear the prerelease flag as the last action of the run.
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

- `release-continuous-deployment`: Narrows the CD trigger to a published
  prerelease, makes the stable Release the output of a passing run rather than
  something published by hand, and extends the published asset set with the
  ResNet-18 package, checksums, and evidence.

## Impact

- `.github/workflows/cd.yml`, `.github/workflows/ci.yml`, and `.github/cd/`.
- `examples/resnet18/` packaging, board entry points, notebook root detection,
  and tests; `examples/matrix-multiplication/` workflow contract tests.
- `docs/rules/ci-cd.md`, the root README, and the v0.1.5 changelog.
- Repository configuration a person must apply: branch protection on `main`
  with `lint-and-simulate` and `release-readiness` required, a
  `pynq-z1-production` environment for the board runner, and a
  `pynq-z1-release` environment whose required reviewers own the decision to
  make a validated candidate stable.
- No RTL, ABI, or numeric change.
