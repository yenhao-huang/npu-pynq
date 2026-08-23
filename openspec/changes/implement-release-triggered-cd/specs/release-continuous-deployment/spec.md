## Purpose

Defines the controlled release-only pipeline that builds, publishes, deploys,
and validates immutable PYNQ-Z1 NPU artifacts without exposing privileged
runners or credentials to ordinary branch activity.

## ADDED Requirements

### Requirement: Published stable release is the only CD trigger
The repository SHALL start production continuous deployment only for a
published, non-draft, non-prerelease GitHub Release whose tag matches
`vMAJOR.MINOR.PATCH`. Pushes to `main`, `dev`, task branches, and tags without a
published Release MUST NOT start production CD.

#### Scenario: Stable release is published
- **WHEN** a `vMAJOR.MINOR.PATCH` GitHub Release is published
- **THEN** exactly one CD workflow run is eligible to build and deploy that release

#### Scenario: Main receives an ordinary push
- **WHEN** a commit is pushed to `main` without publishing a Release
- **THEN** production CD does not start

#### Scenario: Prerelease is published
- **WHEN** a GitHub prerelease is published
- **THEN** production CD does not build or deploy it

### Requirement: Release source is immutable and belongs to main
The CD workflow MUST check out the Release tag and MUST fail before privileged
work when the tag does not resolve to a commit contained in `main` or when the
tag name does not match the release version format.

#### Scenario: Release tag is contained in main
- **WHEN** the release tag resolves to a commit contained in `origin/main`
- **THEN** CD records the full tag commit SHA and may continue to the Vivado job

#### Scenario: Release tag is outside main
- **WHEN** the release tag resolves to a commit not contained in `origin/main`
- **THEN** CD fails before synthesis, artifact publication, or board deployment

### Requirement: Privileged FPGA work uses trusted runners
Synthesis, implementation, timing, and bitstream generation MUST execute only
on a trusted self-hosted runner labeled for Vivado. Physical-board deployment
and validation MUST execute only on a trusted self-hosted runner labeled for
the PYNQ-Z1 and protected by a production environment.

#### Scenario: Release reaches hardware jobs
- **WHEN** release validation succeeds
- **THEN** Vivado and board jobs declare their required trusted runner labels and the board job declares the production environment

### Requirement: Release artifacts are provenance checked
CD SHALL regenerate the `npu_matrix` project, require successful implementation,
timing, and DRC gates, verify the generated BIT/HWH/manifest tuple, and publish
deterministically named overlay and build-evidence assets to the same Release.
No board deployment SHALL start before those checks pass.

#### Scenario: Overlay build and verification pass
- **WHEN** Vivado produces the expected artifact and report directories and provenance verification succeeds
- **THEN** CD publishes the BIT, HWH, manifest, and build evidence for that Release and allows board deployment

#### Scenario: Overlay verification fails
- **WHEN** BIT, HWH, manifest, metadata, hash, timing, or DRC evidence is missing or inconsistent
- **THEN** CD fails and does not deploy the package to the board

### Requirement: Board outcome is objective and retained
The board job SHALL execute the release package non-interactively, SHALL fail
on transfer, provenance, runtime, numeric, or timeout errors, and SHALL retain
machine-readable board evidence as a workflow artifact and Release asset.

#### Scenario: Board example passes
- **WHEN** the deployed release package completes all required matrix cases
- **THEN** CD uploads evidence containing the release tag, commit, overlay provenance, physical limits, per-case metrics, and a PASS marker

#### Scenario: Board example fails
- **WHEN** transfer or any required board case fails
- **THEN** the workflow exits nonzero and does not report successful deployment

### Requirement: Deployment credentials remain external
CD MUST obtain SSH identity and host verification data from the protected
runner or encrypted environment secrets and MUST NOT commit, print, upload, or
store credentials in source, package, workflow artifacts, evidence, or Release
assets.

#### Scenario: Board credentials are unavailable
- **WHEN** the trusted runner has no valid non-interactive board credentials
- **THEN** the board job fails with a credential-availability error without exposing secret material
