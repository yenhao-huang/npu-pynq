## Purpose

Defines deterministic bounded storage planning for exported ResNet graphs and
prevents full im2col or needlessly simultaneous tensor materialization.

## ADDED Requirements

### Requirement: Deterministic tensor lifetimes

The planner SHALL derive each tensor's first definition, final use, byte size,
and required alignment from the validated command order. Given identical graph
and hardware limits, it SHALL produce identical offsets and peak arena size.
Tensor storage MAY be reused only after the prior tensor's final use.

#### Scenario: Non-overlapping lifetimes reuse storage
- **WHEN** two tensors have non-overlapping live intervals and compatible alignment
- **THEN** the deterministic plan may assign them overlapping arena ranges

### Requirement: Live tensor ranges do not alias

Writable output and scratch ranges SHALL not overlap any simultaneously live
input, weight, or metadata range unless an operator explicitly declares and
validates safe in-place behavior. The Phase 2A operator set SHALL default to
out-of-place execution.

#### Scenario: Residual branches remain live
- **WHEN** a residual source is needed after intervening convolution commands
- **THEN** its arena range remains reserved until the residual add consumes it

### Requirement: Bounded lowering scratch

Convolution lowering SHALL allocate scratch only for the current physical input
patch tile, current weight tile, physical signed INT32 result tile, and logical
output-tile accumulator. The plan SHALL NOT allocate a complete im2col tensor.
Scratch byte counts SHALL be checked for overflow and against the declared arena
limit before execution.

#### Scenario: Large spatial convolution
- **WHEN** the logical convolution has many output positions
- **THEN** im2col scratch remains bounded by physical tile limits rather than logical output area

### Requirement: Capacity failure is explicit

The planner and runtime SHALL reject negative, wrapping, misaligned, or
out-of-capacity ranges before allocating or executing model work.

#### Scenario: Arena limit exceeded
- **WHEN** peak live tensors plus required scratch exceed the configured arena limit
- **THEN** preflight fails with the required and available byte counts
