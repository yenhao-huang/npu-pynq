## Purpose

Moves production validation ahead of every Release object. A release branch is
built, packaged and accepted on hardware before anything is tagged, published or
merged, so no Release can exist whose source has not already passed every gate.

## MODIFIED Requirements

### Requirement: A release branch is the only CD trigger
The repository SHALL start production continuous deployment for a push to a
`release/vMAJOR.MINOR.PATCH` branch, and for an explicit manual dispatch on such
a branch. It SHALL NOT start from a published Release, a tag push, `main`, or
`dev`. The branch name MUST equal the version the source declares as
`changelog/vMAJOR.MINOR.PATCH.md`, and that version MUST NOT already be tagged.

#### Scenario: A release branch is pushed
- **WHEN** `release/vMAJOR.MINOR.PATCH` is pushed with a matching declaration
- **THEN** exactly one CD run is eligible to validate it

#### Scenario: Branch name and declaration disagree
- **WHEN** the branch names a version the changelog does not declare
- **THEN** CD fails before any privileged work and names both versions

#### Scenario: The version is already released
- **WHEN** the declared version is already tagged
- **THEN** CD fails rather than rebuilding or republishing it

### Requirement: A passing run produces a draft, never a tag
CD SHALL publish a draft Release from a passing run, holding the standalone
ResNet-18 package, the standalone matrix package, the deterministic overlay
artifacts, one checksum file covering them, and the board evidence from that same
run. CD SHALL NOT create a tag, and SHALL NOT make any Release public. Only the
draft-publishing job MAY hold repository write permission.

#### Scenario: A validation gate fails
- **WHEN** lint, simulation, synthesis, packaging, or board acceptance fails
- **THEN** no draft is produced and nothing is tagged or made public

#### Scenario: A release branch is revalidated
- **WHEN** a corrected release branch is pushed again
- **THEN** the run recreates the draft from the newest validated commit

#### Scenario: Asset set is incomplete
- **WHEN** a package, checksum file, or evidence document is missing
- **THEN** the run fails and publishes no draft

### Requirement: Publication is separate, dispatched, and gated on main
Publishing SHALL be a distinct workflow that is dispatched by hand, builds
nothing, and touches no board. It MUST refuse a Release that is not a draft, a
draft missing any validated asset, a draft whose commit is not contained in
`main`, and a tag that already exists. It SHALL run in an environment whose
reviewers own the release decision.

#### Scenario: The validated commit is not on main
- **WHEN** the promotion was squashed, so the draft's commit is absent from `main`
- **THEN** publication fails and states that a merge commit is required

#### Scenario: The draft is validated and merged
- **WHEN** the draft's commit is contained in `main` and its assets are present
- **THEN** publishing creates the tag and makes that Release public

## ADDED Requirements

### Requirement: Recovery never moves a tag
A failed run SHALL leave no tag and nothing public. A corrected release SHALL be
a new release branch, and a superseded version SHALL NOT be retagged or
republished.

#### Scenario: A run fails after the board produced evidence
- **WHEN** draft publication fails while board evidence already exists
- **THEN** the evidence stays retained on the run and no tag is created
