## MODIFIED Requirements

### Requirement: Signed saturating multiply-accumulate
An enabled PE SHALL interpret both operands as signed INT8 and SHALL register
the exact signed product when both operand-valid inputs are asserted in the
same active step. On the next enabled step, the PE SHALL add each registered
valid product to its signed INT32 accumulator and saturate that update to the
Phase 0 INT32 interval. Products SHALL remain ordered and no accepted product
may be duplicated or lost across a stall.

#### Scenario: Signed endpoints
- **WHEN** an active step accepts any pair drawn from -128, -1, 0, 1, and 127 and a subsequent enabled step occurs
- **THEN** the accumulator changes by the mathematically signed product subject to INT32 saturation

#### Scenario: One operand invalid
- **WHEN** an active step has only one operand-valid input asserted
- **THEN** no product is queued while operand and valid forwarding still follows the pipeline contract

#### Scenario: Accumulator overflow
- **WHEN** a registered valid product would move the accumulator above INT32_MAX or below INT32_MIN
- **THEN** the accumulator clamps to the corresponding signed endpoint without wrapping

### Requirement: Reset clear and global stall
Active-low reset SHALL asynchronously clear the accumulator, registered
product and its validity, forwarded operands, and forwarded valid bits.
Synchronous clear SHALL do the same on the next clock edge and SHALL take
priority over enable. When enable is deasserted, all PE state SHALL hold
exactly.

#### Scenario: Reset during partial work
- **WHEN** active-low reset is asserted while a product is queued or after products have accumulated
- **THEN** accumulator, product, and forwarding state become zero before another active clock edge is required

#### Scenario: Clear while stalled
- **WHEN** synchronous clear is asserted while enable is deasserted
- **THEN** the next clock edge clears accumulator, product, and forwarding state

#### Scenario: Backpressure stall
- **WHEN** enable is deasserted without reset or clear
- **THEN** accumulator, queued product, operands, and valid bits remain unchanged for every stalled edge
