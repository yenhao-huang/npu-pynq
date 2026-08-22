## Purpose

Defines a parameterized output-stationary two-dimensional systolic array with
deterministic skew scheduling, result mapping, and whole-array flow control.

## ADDED Requirements

### Requirement: Rectangular parameterized composition
The array SHALL support positive compile-time row and column counts. It SHALL
instantiate one PE per row-column coordinate, forward A operands to increasing
columns, forward B operands to increasing rows, and expose one signed INT32
accumulator per coordinate in row-major packed order.

#### Scenario: Non-square elaboration
- **WHEN** the array is elaborated with two rows and three columns
- **THEN** it exposes two A edge lanes, three B edge lanes, and six row-major accumulator results

### Requirement: Skewed matrix scheduling
For logical A[M,K] and B[K,N] where M does not exceed array rows and N does not
exceed array columns, software or the upstream controller SHALL inject A[m,k]
at active step m+k and B[k,n] at active step n+k. Unused lanes SHALL have valid
deasserted. The array SHALL accumulate A[m,k]*B[k,n] at coordinate [m,n] on
active step m+n+k.

#### Scenario: Complete logical matrix
- **WHEN** all K reduction elements are injected with the required skew
- **THEN** after active step M+N+K-3 each active accumulator equals the Phase 0 row-major matrix product

#### Scenario: Non-array-aligned logical shape
- **WHEN** M or N is smaller than the physical row or column count
- **THEN** valid masking leaves every unused accumulator at zero and emits no padding result requirement

### Requirement: Whole-array clear and backpressure
One clear input and one enable input SHALL control every PE. Clear SHALL reset
all accumulator and pipeline state on the same edge. A disabled step SHALL
hold the entire array so that upstream inputs can remain stable and the skew
schedule advances only on enabled steps.

#### Scenario: Mid-job stall
- **WHEN** enable is deasserted for one or more clocks during a scheduled matrix job
- **THEN** no operand advances and resuming with the held inputs produces the same result as an unstalled schedule

#### Scenario: Job boundary clear
- **WHEN** clear is asserted between two matrix jobs
- **THEN** the second job result contains no contribution from the first job

### Requirement: Parameter safety
Elaboration SHALL fail or simulation SHALL stop with an explicit diagnostic for
zero rows, zero columns, non-eight-bit operands, or non-32-bit accumulators in
the ABI version 1 implementation.

#### Scenario: Unsupported width
- **WHEN** ABI version 1 RTL is configured with a data width other than 8
- **THEN** validation rejects the configuration before a result is accepted
