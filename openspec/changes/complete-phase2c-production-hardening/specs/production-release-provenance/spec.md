## Purpose

Defines reproducible production release identity, evidence relationships,
transactional promotion, retention, and independently verified rollback.

## ADDED Requirements

### Requirement: Complete immutable release identity
A production release SHALL bind full source commit, clean-source proof, tool
versions, lock or environment identity, BIT/HWH and reports, ABI/capabilities,
model package, corpus, runner sources, target part, target board identity, and
all acceptance evidence through exact byte lengths and SHA-256 digests.

#### Scenario: Tool drift
- **WHEN** an otherwise identical rebuild uses a different unrecorded tool or environment identity
- **THEN** its artifacts cannot reuse the prior release evidence

### Requirement: Evidence topology
The release manifest SHALL distinguish host, simulation, synthesis,
implementation, timing, DRC, board acceptance, soak, and operational evidence;
each claim SHALL be satisfied only by its trusted producer and SHALL reference
its direct inputs without cycles or dangling references.

#### Scenario: Host result in board field
- **WHEN** host integration evidence is supplied as physical-board acceptance
- **THEN** manifest validation fails before promotion

### Requirement: Transactional promotion
Promotion SHALL verify the complete candidate in an immutable versioned
location, atomically update one active selector only after all gates pass, and
leave the prior selector and evidence byte-identical on any failure.

#### Scenario: Evidence upload failure
- **WHEN** inference passes but final evidence persistence or verification fails
- **THEN** the previous active release remains selected

### Requirement: Independently verified rollback
Rollback SHALL re-validate the selected retained release, its evidence graph,
and target compatibility before atomically selecting it. Rollback SHALL never
modify the retained release and SHALL record a new rollback event.

#### Scenario: Retained artifact was modified
- **WHEN** a rollback target differs by one byte from its recorded digest
- **THEN** rollback is rejected and the current release remains active

### Requirement: Retention and revocation
Retention SHALL preserve at least the current and previous verified release,
prevent automatic deletion of an active or rollback target, and support a
signed or equivalently authenticated revocation state that blocks promotion
and rollback to a revoked identity.

#### Scenario: Revoked rollback target
- **WHEN** an operator selects a retained release whose identity is revoked
- **THEN** the selector does not change and diagnostics report revocation
