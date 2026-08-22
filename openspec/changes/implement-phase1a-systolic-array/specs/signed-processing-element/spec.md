## Purpose

Defines one reusable signed processing element whose arithmetic, pipeline
state, reset, and stall behavior match the Phase 0 numeric contract.

## ADDED Requirements

### Requirement: Signed saturating multiply-accumulate
An enabled PE SHALL interpret both operands as signed INT8, form the exact
signed product, add it to its signed INT32 accumulator, and saturate after each
accepted multiply-accumulate to the Phase 0 INT32 interval. It SHALL accumulate
only when both operand-valid inputs are asserted in the same active step.

#### Scenario: Signed endpoints
- **WHEN** an active step accepts any pair drawn from -128, -1, 0, 1, and 127
- **THEN** the accumulator changes by the mathematically signed product subject to INT32 saturation

#### Scenario: One operand invalid
- **WHEN** an active step has only one operand-valid input asserted
- **THEN** the accumulator remains unchanged while operand and valid forwarding still follows the pipeline contract

#### Scenario: Accumulator overflow
- **WHEN** an accepted product would move the accumulator above INT32_MAX or below INT32_MIN
- **THEN** the accumulator clamps to the corresponding signed endpoint without wrapping

### Requirement: Operand and validity forwarding
On every enabled clock edge, the PE SHALL register and forward each operand and
its corresponding validity independently by one pipeline stage. Forwarded
values SHALL preserve signed bit patterns exactly.

#### Scenario: Independent validity
- **WHEN** A is valid and B is invalid on an enabled edge
- **THEN** the next-stage A valid is asserted, B valid is deasserted, and both operand bit patterns are forwarded

### Requirement: Reset clear and global stall
Active-low reset SHALL asynchronously clear the accumulator, forwarded
operands, and forwarded valid bits. Synchronous clear SHALL do the same on the
next clock edge and SHALL take priority over enable. When enable is deasserted,
all PE state SHALL hold exactly.

#### Scenario: Reset during partial work
- **WHEN** active-low reset is asserted after one or more products have accumulated
- **THEN** accumulator and forwarding state become zero before another active clock edge is required

#### Scenario: Clear while stalled
- **WHEN** synchronous clear is asserted while enable is deasserted
- **THEN** the next clock edge clears accumulator and forwarding state

#### Scenario: Backpressure stall
- **WHEN** enable is deasserted without reset or clear
- **THEN** accumulator, operands, and valid bits remain unchanged for every stalled edge
