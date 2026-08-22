# Tiled matrix multiplication

## ADDED Requirements

### Requirement: Logical input contract

The logical multiplier SHALL accept two rank-two NumPy arrays with signed INT8
dtype, positive dimensions, and matching K dimensions. It SHALL reject invalid
types, ranks, dtypes, empty dimensions, mismatched K, non-positive timeouts, and
K larger than the runtime-discovered physical maximum before submitting work.

#### Scenario: K exceeds the physical contract

- **WHEN** K is larger than `runtime.max_k`
- **THEN** validation fails before any call to `runtime.run`
- **AND** the implementation does not approximate the result with K tiling

### Requirement: M/N edge tiling

The logical multiplier SHALL partition M and N by the runtime-discovered
physical limits and submit every tile through `runtime.run` with dense
C-contiguous signed INT8 operands.

#### Scenario: Both output dimensions have edge tiles

- **WHEN** a 3x5 matrix is multiplied by a 5x3 matrix on a 2x2 physical array
- **THEN** exactly four physical jobs are submitted with output shapes 2x2,
  2x1, 1x2, and 1x1
- **AND** the assembled signed INT32 result equals the NumPy reference

### Requirement: One finite logical deadline

The multiplier SHALL apply one positive software timeout to the complete
logical multiplication and pass only the remaining time to each physical job.

#### Scenario: Deadline expires between tiles

- **WHEN** the deadline is exhausted after a completed tile
- **THEN** the multiplier raises `TimeoutError` before submitting the next tile

### Requirement: Repeatability and ownership

Each call SHALL return an owned C-contiguous signed INT32 result and SHALL not
reuse stale output from a previous call.

#### Scenario: Repeated execution changes operands

- **WHEN** the same multiplier runs two jobs with different operands
- **THEN** each result independently matches its reference

### Requirement: Performance accounting

The multiplier SHALL report immutable logical dimensions, physical tile count,
elapsed seconds, MAC count, operation count, and operations per second.

#### Scenario: Nonzero elapsed time

- **WHEN** a job with dimensions M, N, and K completes in positive elapsed time
- **THEN** MAC count is `M*N*K`
- **AND** operation count is `2*M*N*K`
- **AND** throughput is operation count divided by elapsed seconds
