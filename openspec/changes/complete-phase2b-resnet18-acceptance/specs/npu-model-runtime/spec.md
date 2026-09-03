## Purpose

Extends model execution with observable physical cycles and bounded layer
capture required for Phase 2B acceptance.

## ADDED Requirements

### Requirement: Physical cycle telemetry

After every successful physical job, the Phase 1 runtime SHALL expose a stable
cycle-counter value without changing its owned NumPy result contract. Matrix
and model metrics SHALL sum cycles only when every contributing job exposes
compatible telemetry; otherwise cycle sum SHALL be explicitly unavailable.

#### Scenario: Counter rollover during read
- **WHEN** the high cycle word changes while the low word is read
- **THEN** the runtime retries within the existing deadline and never reports a torn value

### Requirement: Bounded immutable tensor capture

Model execution SHALL optionally capture only declared, produced activation
tensors requested before execution. Captures SHALL be owned C-contiguous INT8
arrays in an immutable mapping and SHALL not alias the reusable arena.

#### Scenario: Unknown capture request
- **WHEN** a requested capture name is not a produced activation tensor
- **THEN** input preflight fails before any physical call
