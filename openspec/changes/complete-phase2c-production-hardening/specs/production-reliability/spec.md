## Purpose

Defines repeatable long-running regression, recovery, drift, and resource
stability evidence required before an NPU artifact is production-qualified.

## ADDED Requirements

### Requirement: Changed-input soak acceptance
The production runner SHALL execute a declared, content-addressed sample
schedule for a declared invocation count and minimum duration, prevent a
single repeated tensor from satisfying the schedule, and compare every result
to its exact accepted reference.

#### Scenario: Stale success loop
- **WHEN** a soak schedule repeatedly submits one unchanged input despite declaring multiple samples
- **THEN** soak acceptance fails as invalid evidence even if every output matches

### Requirement: Long-running stability gates
Soak evidence SHALL report completed and failed invocations, recovery count,
latency and cycle distributions, throughput, logical traffic, physical work,
process memory, allocator or buffer counts, temperature when available, and
first-to-last drift. Declared absolute and slope limits SHALL be enforced.

#### Scenario: Memory growth
- **WHEN** owned process memory grows beyond the configured absolute or per-invocation limit
- **THEN** the run fails without promoting its release or replacing known-good evidence

### Requirement: Recovery campaign
The runner SHALL execute declared failures at deterministic schedule points,
prove that each failed invocation publishes no output, verify bounded reset,
and re-run a changed known-good sample exactly after recovery.

#### Scenario: Recovery returns stale output
- **WHEN** the post-recovery result equals a prior sample instead of the changed recovery sample
- **THEN** production reliability acceptance fails and records the failing schedule point

### Requirement: Reproducible regression evidence
Evidence SHALL bind schedule, seed, corpus order, thresholds, runner source,
clock source, warmup policy, sample count, and exact invocation count. An
interrupted or partial run SHALL never be labeled complete.

#### Scenario: Interrupted soak
- **WHEN** the process exits before the declared duration and invocation gates both pass
- **THEN** only failure diagnostics may be retained and no success evidence is published
