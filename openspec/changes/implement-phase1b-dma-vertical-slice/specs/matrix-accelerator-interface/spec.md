## Purpose

Defines the software-visible control and streaming behavior that turns the
Phase 1A systolic array into a recoverable DMA-connected matrix accelerator.

## ADDED Requirements

### Requirement: ABI v2 control window
The accelerator SHALL expose one 32-bit little-endian AXI4-Lite window with the
ABI v2 identity, capability, control, status, error, dimension, stride,
timeout, cycle-counter, job-flag, and output-zero-point registers. JOB_FLAGS
at 0x3C identifies first and final K slices; OUTPUT_ZERO_POINT is at 0x40.
Reserved offsets from 0x44 through 0xFF SHALL read zero and ignore writes. AXI write-address and
write-data channels SHALL be accepted independently, and every accepted read or
write SHALL produce exactly one stable response under response backpressure.

#### Scenario: Independent write channels
- **WHEN** AWVALID and WVALID arrive on different cycles for a writable register
- **THEN** the write commits once after both handshakes and produces one OKAY response

#### Scenario: Reserved access
- **WHEN** software reads or writes a reserved ABI v2 offset
- **THEN** the read returns zero and the write has no externally visible effect

### Requirement: Bounded physical job validation
The Phase 1B implementation SHALL accept dense jobs with M and N in [1,2], K in
[1,256], A_STRIDE equal to K, B_STRIDE equal to N, C_STRIDE equal to N, and a
nonzero TIMEOUT_CYCLES value. Unsupported dimensions SHALL fail before accepting
stream data with INVALID_DIMENSION; unsupported strides SHALL fail with
INVALID_STRIDE; and a zero timeout SHALL fail with INVALID_TIMEOUT. The physical
limits SHALL be parameters visible in generated HWH metadata so runtime can
reject unsupported jobs before START. The ABI-wide 16-bit dimension encoding is
unchanged; Phase 2 tiling will map larger logical matrices to bounded physical
jobs.

#### Scenario: Valid physical job
- **WHEN** software starts a dense M=2 N=2 K=256 job with a nonzero timeout
- **THEN** BUSY sets and the accelerator becomes ready for the A input frame

#### Scenario: Shape exceeds physical array
- **WHEN** software starts a job with M=3 on the 2-row implementation
- **THEN** no stream element is accepted and ERROR reports INVALID_DIMENSION

### Requirement: Two-frame matrix input stream
After an accepted START, the accelerator SHALL consume exactly M*K signed INT8
A elements in row-major order as one AXI4-Stream frame, followed by exactly K*N
signed INT8 B elements in row-major order as a second frame. Each input beat
SHALL carry one element in an 8-bit TDATA value. TLAST SHALL be asserted only on
the final accepted beat of each frame. TVALID data and phase state SHALL remain
stable while TREADY is deasserted.

#### Scenario: Correct A and B frames
- **WHEN** A and B each present the declared element count and TLAST on their final accepted beats
- **THEN** the controller begins the matrix computation with the exact signed elements received

#### Scenario: Early or missing TLAST
- **WHEN** TLAST is accepted before the declared final element or is absent on that element
- **THEN** BUSY clears, DONE remains clear, ERROR reports STREAM_LENGTH, and no successful output frame is produced

### Requirement: Hardware requantized matrix output stream
For a valid job, the accelerator SHALL use the Phase 1A signed INT8, per-MAC
saturating INT32, row-major matrix contract. It SHALL accumulate ordered K
slices in hardware, consume final-slice signed INT32 bias and Q1.31 multiplier
frames plus unsigned shift values, apply bias and requantization once, and
produce exactly M*N signed INT8 results on an 8-bit AXI4-Stream output. It SHALL assert
TLAST only with the final result. TDATA, TVALID, and TLAST SHALL remain stable
until each beat is accepted.

#### Scenario: Backpressured result
- **WHEN** the output consumer deasserts TREADY while a result is valid
- **THEN** the same result and TLAST value remain asserted until the handshake occurs

#### Scenario: Non-square logical mask
- **WHEN** a valid M=1 N=2 job executes on the physical 2x2 array
- **THEN** the output contains exactly two logical row-major values and no padded lane

### Requirement: Recoverable job lifecycle
START SHALL be write-one and accepted only while idle; SOFT_RESET SHALL be
write-one and take priority over active state. BUSY SHALL cover input,
computation, and output. DONE and ERROR SHALL be sticky until a newly accepted
START or SOFT_RESET. The first error code SHALL remain latched. A START while
BUSY SHALL preserve the active job, latch BUSY_START, and allow that job's data
path to continue. The 64-bit cycle counter SHALL count BUSY cycles and remain
stable after completion. Reaching TIMEOUT_CYCLES before the final output
handshake SHALL terminate the job with TIMEOUT.

#### Scenario: Successful completion
- **WHEN** the final output beat of an error-free job handshakes before timeout
- **THEN** BUSY clears, DONE sets, ERROR stays clear, and the cycle counter becomes stable

#### Scenario: Soft reset during a job
- **WHEN** SOFT_RESET is written while BUSY is set
- **THEN** active input, compute, and output state is abandoned and all status, error, dimensions, and cycle state return to reset values
