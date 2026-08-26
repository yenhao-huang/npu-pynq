## Purpose

Defines fail-closed compatibility, reset, timeout, and failure behavior for a
production NPU across package, runtime, overlay, and recovery boundaries.

## ADDED Requirements

### Requirement: Dependency acceptance gate
Production hardening SHALL accept a baseline only when the Phase 2B artifact
and board evidence are immutable, digest-matched, trusted, merged, and identify
the same source and target board.

#### Scenario: Synthetic baseline
- **WHEN** the only available Phase 2B result is host or synthetic evidence
- **THEN** production qualification remains blocked before release promotion

### Requirement: Explicit compatibility negotiation
The runtime SHALL reject ABI-major mismatches, missing mandatory capabilities,
unsupported limits, unknown required feature bits, and incompatible evidence
schemas before programming or submitting work. A newer ABI minor SHALL be
accepted only when all declared requirements remain supported.

#### Scenario: Unknown required capability
- **WHEN** a package requires a capability the physical runtime does not expose
- **THEN** preflight returns a typed compatibility failure with no DMA or MMIO job start

### Requirement: Deterministic reset and bounded recovery
Reset SHALL clear command state, partial results, counters that are defined as
per-run, outstanding interrupts, and reused activation storage into a declared
idle state within a bounded deadline. Recovery SHALL either restore that state
or mark the device unavailable without silently retrying non-idempotent work.

#### Scenario: Timeout followed by recovery
- **WHEN** a job times out after partial physical activity
- **THEN** the failed invocation publishes no result and the next invocation begins only after verified idle recovery

### Requirement: Typed failure contract
Every validation, compatibility, timeout, DMA, hardware, model-command,
evidence, and recovery failure SHALL expose a stable category, bounded detail,
operation context, and retryability classification without leaking secrets or
raw customer tensors.

#### Scenario: Model command failure
- **WHEN** a physical failure occurs during a lowered convolution
- **THEN** diagnostics identify the command and stable failure category but omit tensor contents and credentials
