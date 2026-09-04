## Purpose

Defines standalone, provenance-bound PYNQ-Z1 delivery and trusted hardware
evidence for complete ResNet-18 acceptance.

## ADDED Requirements

### Requirement: Verified standalone package

The package builder SHALL copy only an explicit allowlist of runtime and
example sources, include expected file digests, exclude host paths and
credentials, and reproduce byte-identical archives from identical inputs.

#### Scenario: Unexpected source file
- **WHEN** packaging discovers a file outside the allowlist
- **THEN** it fails before producing a final archive

### Requirement: Provenance before execution

The board runner SHALL verify source commit, archive, BIT, HWH, Phase 2A
package, corpus, ABI, capabilities, and physical limits before the first model
operation.

#### Scenario: Overlay mismatch
- **WHEN** the BIT or HWH digest differs from the acceptance descriptor
- **THEN** board execution fails before overlay programming or model submission

### Requirement: Complete trusted evidence

Board acceptance evidence SHALL bind clean synthesis, implementation, routed
timing, DRC, utilization, accuracy, latency, throughput, bandwidth, cycles,
repeatability, recovery, environment, and every input digest. Host or simulated
results SHALL NOT satisfy a physical-board field.

#### Scenario: Board is unavailable
- **WHEN** the protected PYNQ-Z1 runner cannot be reached
- **THEN** the gate remains explicitly blocked and no board-pass evidence is emitted

### Requirement: Atomic promotion and rollback

Deployment SHALL retain immutable versioned directories and change the active
target only after validation succeeds. Rollback SHALL select a previously
verified version without modifying its contents.

#### Scenario: Mid-run deployment failure
- **WHEN** packaging, transfer, verification, or inference fails
- **THEN** the prior active deployment and evidence remain selected
