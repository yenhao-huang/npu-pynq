## Purpose

Defines the downloadable ResNet-18 package that lets a user with a PYNQ-Z1 and
no development tools classify a real photograph, and the non-interactive board
acceptance that proves the same package on hardware before it is published.

## ADDED Requirements

### Requirement: The package is complete and self-describing
The standalone package SHALL carry the notebook, the shared runtime and export
sources it imports, the quantized model workspace, the ImageNet class list, the
bundled photographs, the demo input tensor with its host record, the verified
BIT and HWH artifacts, and the Vivado build reports. It SHALL carry a manifest
recording the release tag, the source commit, the overlay digests, the Vivado
timing and DRC gates, and the SHA-256 of every file. Identical inputs SHALL
produce a byte-identical archive, and no build-host path SHALL appear in it.

#### Scenario: A required input is missing
- **WHEN** an artifact, report, acceptance asset, or model workspace file is absent
- **THEN** packaging fails and publishes no archive

#### Scenario: The overlay failed a Vivado gate
- **WHEN** build evidence reports DRC errors, failing setup paths, non-positive slack, or another source commit
- **THEN** packaging fails and publishes no archive

#### Scenario: The same inputs are packaged twice
- **WHEN** packaging runs twice over identical inputs
- **THEN** both archives are byte-identical

### Requirement: The board proves the package before it programs the overlay
Board acceptance SHALL verify the deployed tree against the published archive
digest and against every per-file digest in the package manifest, SHALL reject
any file the manifest does not declare other than generated evidence, and SHALL
verify that the deployed overlay matches the manifest, all before the bitstream
is programmed or any model operation runs.

#### Scenario: The archive differs from the published asset
- **WHEN** the deployed archive digest differs from the expected digest
- **THEN** acceptance fails before overlay programming

#### Scenario: The deployed tree gained a file
- **WHEN** the extracted tree contains a file the manifest does not declare
- **THEN** acceptance fails and names that file

### Requirement: Real-image acceptance runs on the physical board
Non-interactive real-image acceptance SHALL run the host-prepared input tensor
on the physical overlay, SHALL compare every capture against the host record
exactly, SHALL fail when a declared expected class does not win, and SHALL fail
when no physical job ran. A host backend SHALL NOT be able to produce this
evidence.

#### Scenario: The board disagrees with the host
- **WHEN** any board capture digest differs from the host record
- **THEN** acceptance fails and publishes no evidence

#### Scenario: No physical job ran
- **WHEN** the run reports zero physical jobs
- **THEN** acceptance fails rather than reporting a board pass

### Requirement: Board evidence identifies what ran
Published board evidence SHALL name the release tag, the source commit, the
overlay digests and target part, the model digests, the input image with its
provenance and digest, the top-5 prediction, and the verdict. Evidence SHALL be
written once, atomically, and only after every gate passes.

#### Scenario: A gate fails mid-run
- **WHEN** any acceptance gate fails
- **THEN** no evidence file is written and any previous evidence is untouched

### Requirement: Deployment promotes only after readable evidence
Deployment SHALL stage the package in a run-specific directory, run acceptance
there, make the evidence readable, and only then move the staged directory to
its immutable versioned destination and update the active selection.

#### Scenario: Acceptance fails on the board
- **WHEN** the board run fails before evidence exists
- **THEN** the previous deployment and its evidence remain selected
