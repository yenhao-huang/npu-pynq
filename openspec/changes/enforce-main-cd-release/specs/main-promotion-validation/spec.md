## Purpose

Defines the enforced gate between an approved promotion and `main`, so that
continuous deployment always knows which release a validated commit publishes
and never validates a commit that skipped review.

## ADDED Requirements

### Requirement: Promotion into main declares its release version
A pull request targeting `main` SHALL declare the version it will release as
`changelog/vMAJOR.MINOR.PATCH.md`, and the pre-merge gate SHALL fail when the
declared version is absent, empty, unparsable, or already tagged. The promotion
SHALL be merged with a merge commit, so the validated commit becomes contained in
`main`.

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
Cutting the release branch, deploying to the physical board, merging into
`main`, and publishing the draft SHALL each be separately approved. Board
deployment SHALL use a protected environment, and publication SHALL use an
environment whose reviewers own the release decision. Code review approval SHALL
NOT imply any of them.

#### Scenario: Validation passes but nobody publishes
- **WHEN** every gate passes and the draft is not published
- **THEN** the release stays a validated draft with no tag

### Requirement: The pre-merge gate protects the deployment path itself
The pre-merge gate SHALL fail when continuous deployment no longer runs on a
release branch, when its draft-publishing job no longer depends on every
validation job, when any other job gains repository write permission, or when
publication is no longer a dispatched workflow gated by the release environment.

#### Scenario: A change weakens the deployment gate
- **WHEN** a pull request removes a validation dependency from the draft job
- **THEN** the pre-merge gate fails before that change can reach `main`
