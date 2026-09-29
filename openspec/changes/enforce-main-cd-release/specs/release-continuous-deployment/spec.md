## Purpose

Redefines production continuous deployment as the validation that precedes a
release rather than the work that follows one, so that no published Release
exists whose source has not already passed every host, synthesis, and physical
board gate.

## MODIFIED Requirements

### Requirement: Validated main commit is the only CD trigger
The repository SHALL start production continuous deployment for every push to
`main` and for an explicit manual dispatch, and SHALL NOT start it from a
published GitHub Release, a tag push, or activity on any other branch. A run
SHALL proceed past version selection only when the commit declares a release
version that is not yet tagged.

#### Scenario: Approved promotion lands on main
- **WHEN** a commit reaches `main` declaring a not-yet-tagged version
- **THEN** exactly one CD run is eligible to validate and publish that version

#### Scenario: Main receives a commit declaring no new version
- **WHEN** a commit reaches `main` whose declared version is already tagged
- **THEN** the run reports that it publishes nothing and performs no privileged work

#### Scenario: A Release is published by hand
- **WHEN** a GitHub Release is published outside this workflow
- **THEN** production CD does not start, and that Release carries no CD evidence

### Requirement: Release publication follows every validation gate
The workflow SHALL create the release tag and publish the GitHub Release only
after host checks, the Vivado build, package assembly, and physical PYNQ-Z1
acceptance have all succeeded for the same commit. Only the publishing job MAY
hold repository write permission, and it MUST refuse a tag that already exists
or a commit no longer contained in `origin/main`.

#### Scenario: A validation gate fails
- **WHEN** lint, simulation, synthesis, packaging, or board acceptance fails
- **THEN** no tag is created and no GitHub Release is published

#### Scenario: The declared tag already exists
- **WHEN** publication starts for a tag that is already present on the remote
- **THEN** the run fails instead of moving, deleting, or republishing that tag

#### Scenario: Every gate passes
- **WHEN** all validation jobs succeed for one commit
- **THEN** that commit is tagged and released with its packages, checksums, and evidence

### Requirement: Release assets carry the deliverable and its evidence
A published Release SHALL carry the standalone ResNet-18 package, the
standalone matrix package, the deterministic overlay artifacts, one checksum
file covering them, and the board evidence produced by the same run, including
the real-image acceptance result.

#### Scenario: Asset set is incomplete
- **WHEN** a package, checksum file, or evidence document is missing before publication
- **THEN** publication fails and no partial Release is created

#### Scenario: Checksums disagree with the assets
- **WHEN** a downloaded package digest differs from the recorded checksum
- **THEN** the run fails before the board is deployed to and before publication

## ADDED Requirements

### Requirement: Recovery is a new version, never a moved tag
A failed run SHALL leave the repository unchanged. Republishing a version
SHALL NOT be possible; a corrected release SHALL be declared as a new version.

#### Scenario: A run fails after the board accepted the package
- **WHEN** publication fails while board evidence already exists
- **THEN** the evidence remains retained on the run and the next validated commit publishes
