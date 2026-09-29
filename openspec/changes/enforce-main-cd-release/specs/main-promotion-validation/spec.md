## Purpose

Defines the enforced gate between an approved promotion and `main`, so that
continuous deployment always knows which release a validated commit publishes
and never validates a commit that skipped review.

## ADDED Requirements

### Requirement: Promotion into main declares its release version
A pull request targeting `main` SHALL declare the version it will release as
`changelog/vMAJOR.MINOR.PATCH.md`, and the pre-merge gate SHALL fail when the
declared version is absent, empty, unparsable, or already tagged.

#### Scenario: Promotion declares a new version
- **WHEN** a pull request into `main` adds a changelog file for an untagged version
- **THEN** the pre-merge gate passes and continuous deployment can select that version

#### Scenario: Promotion declares nothing
- **WHEN** a pull request into `main` adds no new version declaration
- **THEN** the pre-merge gate fails and names the file that must be added

### Requirement: Pre-merge checks cover every commit reaching main
Open-source lint, simulation, and the host model and example suites SHALL run
for every pull request into `main`, and SHALL be required checks so that no
commit reaches `main` without them.

#### Scenario: Host suites fail on a promotion
- **WHEN** lint, simulation, or a host suite fails on a pull request into `main`
- **THEN** the promotion cannot merge

### Requirement: Approval boundaries are explicit
Merging into `main`, publishing the candidate prerelease, deploying to the
physical board, and promoting the candidate to a stable Release SHALL each be
separately approved. Board deployment SHALL use a protected environment, and
promotion SHALL use an environment whose reviewers own the release decision.
Code review approval SHALL NOT imply any of them.

#### Scenario: Validation passes but nobody approves promotion
- **WHEN** every gate passes and the promotion environment is not approved
- **THEN** the candidate stays a prerelease and the run waits

### Requirement: The pre-merge gate protects the deployment path itself
The pre-merge gate SHALL fail when the continuous deployment workflow no longer
runs on a published release, when its promotion job no longer depends on every
validation job, or when any other job gains repository write permission.

#### Scenario: A change weakens the deployment gate
- **WHEN** a pull request removes a validation dependency from the promoting job
- **THEN** the pre-merge gate fails before that change can reach `main`
