## Purpose

Defines the bit-accurate arithmetic and layout contract shared by NPU RTL,
software export, runtime execution, and differential verification.

## ADDED Requirements

### Requirement: Signed integer domains
Matrix operands SHALL be signed two's-complement INT8 values in the inclusive
range -128 through 127. Products SHALL be mathematically exact signed INT16
values, and accumulator-visible results SHALL be signed INT32 values.

#### Scenario: Signed endpoint multiplication
- **WHEN** an operand pair contains any combination of -128, -1, 0, 1, and 127
- **THEN** its product equals mathematical signed multiplication without unsigned reinterpretation

### Requirement: Deterministic accumulation and overflow
Each multiply-accumulate SHALL add the exact signed product to the prior INT32
accumulator and saturate the result to [-2147483648, 2147483647]. Accumulation
SHALL occur in increasing reduction-index order, and reset SHALL restore zero
before the next accepted product.

#### Scenario: Positive accumulator overflow
- **WHEN** an addition would produce a value greater than 2147483647
- **THEN** the observable accumulator value is 2147483647

#### Scenario: Negative accumulator overflow
- **WHEN** an addition would produce a value less than -2147483648
- **THEN** the observable accumulator value is -2147483648

#### Scenario: Reset between jobs
- **WHEN** reset is asserted before a new matrix job accepts its first product
- **THEN** no accumulator value from the preceding job contributes to the new result

### Requirement: Matrix and tensor layout
The canonical matrix contract SHALL use dense row-major A[M,K], B[K,N], and
C[M,N] arrays. The canonical neural-network activation layout SHALL be NHWC,
and convolution weights SHALL be HWIO. Dimensions, element strides, and byte
strides SHALL be positive and SHALL not alias writable output with an input.

#### Scenario: Matrix reference indexing
- **WHEN** logical element A[m,k], B[k,n], or C[m,n] is addressed
- **THEN** its dense offset is respectively m*K+k, k*N+n, or m*N+n

#### Scenario: Invalid aliasing
- **WHEN** an output range overlaps either input range for the same job
- **THEN** the runtime rejects the job before hardware execution

### Requirement: Requantization rounding and saturation
INT32 to INT8 requantization SHALL compute an INT64 intermediate from the
accumulator and a signed Q1.31 multiplier, divide by 2^(31+shift), round to the
nearest integer with exact ties away from zero, add the signed INT8 output zero
point, and saturate to [-128,127]. The shift SHALL be in [0,31].

#### Scenario: Positive halfway value
- **WHEN** the scaled intermediate is exactly halfway between positive integers
- **THEN** requantization selects the integer farther from zero

#### Scenario: Negative halfway value
- **WHEN** the scaled intermediate is exactly halfway between negative integers
- **THEN** requantization selects the integer farther from zero

#### Scenario: INT8 output saturation
- **WHEN** the rounded and zero-point-adjusted value lies outside [-128,127]
- **THEN** the result is clamped to the nearest signed INT8 endpoint
