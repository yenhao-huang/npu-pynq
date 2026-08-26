## Purpose

Defines bounded, privacy-preserving production health and diagnostic artifacts
that operators can use to distinguish compatibility, workload, and device faults.

## ADDED Requirements

### Requirement: Stable health snapshot
The system SHALL expose a versioned snapshot containing release identity,
runtime state, ABI/capabilities/limits, reset and recovery state, last stable
failure category, bounded counters, and availability without performing a model
operation or mutating device state.

#### Scenario: Health query during failure
- **WHEN** the device is unavailable after failed recovery
- **THEN** the snapshot reports unavailable and the stable cause without starting DMA or clearing the failure

### Requirement: Bounded failure ledger
The runtime SHALL maintain a bounded chronological ledger of state transitions,
jobs, timeouts, typed failures, resets, recoveries, promotions, rollbacks, and
revocations. Entries SHALL include stable monotonic ordering and release
identity but exclude raw tensors, credentials, private paths, and unbounded text.

#### Scenario: Ledger capacity reached
- **WHEN** a new event arrives after the configured ledger capacity is full
- **THEN** the oldest entry is evicted deterministically and the dropped-entry counter increments

### Requirement: Secret-safe support bundle
An operator SHALL be able to export a canonical, content-addressed support
bundle containing health, bounded ledger, sanitized environment, and referenced
public evidence. Export SHALL apply an allowlist and fail if an unexpected file,
environment key, credential marker, absolute private path, or tensor payload is present.

#### Scenario: Credential-like field
- **WHEN** a diagnostic source includes an unallowlisted token or private-key field
- **THEN** support-bundle creation fails without publishing a partial archive

### Requirement: Diagnostic compatibility
Diagnostic schemas SHALL use explicit major/minor versions, reject unknown
required fields or incompatible majors, and preserve unknown optional minor
extensions without treating them as proof of a required gate.

#### Scenario: New optional counter
- **WHEN** a newer minor snapshot adds an optional bounded counter
- **THEN** an older compatible consumer may retain it but cannot use it to satisfy an unknown requirement
