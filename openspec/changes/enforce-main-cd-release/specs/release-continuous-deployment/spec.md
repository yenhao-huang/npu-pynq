## Purpose

Keeps a published release as the trigger for production continuous deployment,
but makes the published stable Release the result of that validation rather than
its precondition, so that no stable Release exists whose source has not passed
every host, synthesis, and physical board gate.

## MODIFIED Requirements

### Requirement: A published prerelease is the only CD trigger
The repository SHALL start production continuous deployment only for a
published, non-draft **prerelease** whose tag matches `vMAJOR.MINOR.PATCH`, and
for an explicit manual dispatch naming such a prerelease. A draft, a tag in
another format, and a Release that is already stable SHALL each be refused
before any privileged work.

#### Scenario: A release candidate is published as a prerelease
- **WHEN** a `vMAJOR.MINOR.PATCH` prerelease is published
- **THEN** exactly one CD run is eligible to validate and promote it

#### Scenario: A stable Release is published by hand
- **WHEN** a Release is published that is already stable
- **THEN** CD fails and states that the candidate must be published as a prerelease

#### Scenario: A prerelease becomes stable
- **WHEN** the prerelease flag is cleared at the end of a passing run
- **THEN** no further CD run starts, because that transition is not a publication

### Requirement: The candidate's source is immutable and belongs to main
CD MUST resolve the prerelease tag to a commit contained in `origin/main`, and
MUST fail before privileged work when it is not, or when the tag differs from
the version that commit declares as `changelog/vMAJOR.MINOR.PATCH.md`.

#### Scenario: Candidate tag is outside main
- **WHEN** the tag resolves to a commit not contained in `origin/main`
- **THEN** CD fails before synthesis, packaging, or board deployment

#### Scenario: Candidate tag disagrees with the declared version
- **WHEN** the prerelease tag is not the version the source declares
- **THEN** CD fails and names both versions

### Requirement: Promotion follows every validation gate
The workflow SHALL attach the packages, overlay artifacts, checksums, and board
evidence to the candidate and clear its prerelease flag only after host checks,
the Vivado build, package assembly, and physical PYNQ-Z1 acceptance have all
succeeded for that tag. Only the promoting job MAY hold repository write
permission.

#### Scenario: A validation gate fails
- **WHEN** lint, simulation, synthesis, packaging, or board acceptance fails
- **THEN** the candidate remains a prerelease and no stable Release is produced

#### Scenario: Every gate passes
- **WHEN** all validation jobs succeed for one candidate
- **THEN** its assets are attached and only then is the prerelease flag cleared

### Requirement: Release assets carry the deliverable and its evidence
A promoted Release SHALL carry the standalone ResNet-18 package, the standalone
matrix package, the deterministic overlay artifacts, one checksum file covering
them, and the board evidence produced by the same run, including the real-image
acceptance result.

#### Scenario: Asset set is incomplete
- **WHEN** a package, checksum file, or evidence document is missing before promotion
- **THEN** promotion fails and the candidate remains a prerelease

#### Scenario: Checksums disagree with the assets
- **WHEN** a downloaded package digest differs from the recorded checksum
- **THEN** the run fails before the board is deployed to and before promotion

## ADDED Requirements

### Requirement: A failed candidate stays a prerelease
A failed run SHALL leave the candidate as a prerelease, which is the record that
it was not accepted. The prerelease flag SHALL NOT be cleared by hand to finish a
run, and a corrected release SHALL be a new version rather than a moved tag.

#### Scenario: A run fails after the board produced evidence
- **WHEN** promotion fails while board evidence already exists
- **THEN** the evidence stays retained on the run and the candidate stays a prerelease
