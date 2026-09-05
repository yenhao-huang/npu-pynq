## Purpose

Defines a versioned software-visible contract for capability discovery, matrix
job control, streaming order, status, errors, and forward compatibility.

## ADDED Requirements

### Requirement: ABI identity and compatibility
The control interface SHALL expose magic value 0x3155504E (ASCII "NPU1" in
little-endian register order), ABI major and minor versions, and capability
bits. Software SHALL reject a different magic value or unsupported major
version and SHALL tolerate a newer minor version when all required capability
bits are present.

#### Scenario: Compatible minor version
- **WHEN** hardware reports ABI major 2, a newer minor version, and all capabilities required by the job
- **THEN** version negotiation succeeds without assuming unadvertised behavior

#### Scenario: Incompatible major version
- **WHEN** hardware reports a major version other than 2
- **THEN** software rejects the device before submitting a job

### Requirement: Stable control register map
ABI version 2 SHALL use a 32-bit little-endian AXI4-Lite register window with
the following byte offsets: MAGIC 0x00, VERSION 0x04, CAPABILITIES 0x08,
CONTROL 0x0C, STATUS 0x10, ERROR 0x14, M 0x18, N 0x1C, K 0x20, A_STRIDE 0x24,
B_STRIDE 0x28, C_STRIDE 0x2C, TIMEOUT_CYCLES 0x30, CYCLES_LO 0x34,
CYCLES_HI 0x38, JOB_FLAGS 0x3C, OUTPUT_ZERO_POINT 0x40, and RESERVED from
0x44 through 0xFF. Reserved locations SHALL
read as zero and ignore writes.

#### Scenario: Reserved register access
- **WHEN** software reads or writes a reserved ABI version 2 register
- **THEN** the read returns zero and the write has no externally visible effect

### Requirement: Job control state machine
CONTROL bit 0 SHALL be a write-one START request and bit 1 SHALL be a write-one
SOFT_RESET request. STATUS bit 0 SHALL indicate BUSY, bit 1 DONE, and bit 2
ERROR. DONE and ERROR SHALL remain set until a new accepted START or
SOFT_RESET. A START while BUSY SHALL not modify the active job and SHALL set
the BUSY_START error.

#### Scenario: Successful job lifecycle
- **WHEN** a valid START is accepted while idle and the declared matrix output completes
- **THEN** BUSY clears, DONE sets, ERROR stays clear, and the 64-bit cycle counter is stable

#### Scenario: Start while busy
- **WHEN** START is written while BUSY is set
- **THEN** the active job continues unchanged and ERROR reports BUSY_START

#### Scenario: Soft reset
- **WHEN** SOFT_RESET is written
- **THEN** active state is abandoned, status and error latches clear, and matrix dimensions return to zero

### Requirement: Matrix stream transaction
For a valid MxK by KxN job, the input stream SHALL consume exactly M*K A
elements followed by K*N B elements, both in canonical row-major order.
Non-final K slices SHALL retain ordered INT32 partial sums. The final slice
SHALL additionally consume signed INT32 bias and multiplier frames and an
unsigned shift frame, then produce exactly M*N signed INT8 C elements after
one Q1.31 requantization. Each transfer SHALL use little-endian element bytes and TLAST
SHALL mark the final element of its logical stream.

#### Scenario: Non-tile-aligned dimensions
- **WHEN** M, N, or K is not a multiple of the implementation tile dimension
- **THEN** the transaction transfers only declared logical elements and produces no padding elements

#### Scenario: Early TLAST
- **WHEN** TLAST arrives before the declared final input element
- **THEN** the job stops with STREAM_LENGTH error and produces no successful DONE state

### Requirement: DMA buffer contract
DMA buffer base addresses SHALL be aligned to 64 bytes and SHALL identify a
non-wrapping range within the 32-bit physical address space. Allocated sizes
SHALL cover the declared dense payload: M*K bytes for A, K*N bytes for B, and
M*N bytes for C. The writable C range SHALL not overlap either input range.

#### Scenario: Misaligned buffer
- **WHEN** any matrix buffer base address is not divisible by 64
- **THEN** the runtime rejects the job before configuring DMA or starting hardware

#### Scenario: Undersized output buffer
- **WHEN** the C allocation contains fewer than M*N bytes
- **THEN** the runtime rejects the job before hardware execution

#### Scenario: Output aliases an input
- **WHEN** the writable C physical range overlaps the A or B physical range
- **THEN** the runtime rejects the job before hardware execution

### Requirement: Job validation and error codes
M, N, and K SHALL each be in [1,65535], byte strides SHALL cover one logical
row and be multiples of the element size, and TIMEOUT_CYCLES SHALL be nonzero.
The first detected failure SHALL latch one of NONE=0, INVALID_DIMENSION=1,
INVALID_STRIDE=2, BUSY_START=3, STREAM_LENGTH=4, TIMEOUT=5,
INVALID_TIMEOUT=6, INVALID_REQUANTIZATION=7, or INTERNAL=255. Software preflight validation SHALL expose
the corresponding error code together with its diagnostic message.

#### Scenario: Invalid dimension
- **WHEN** START is requested with any zero dimension
- **THEN** no stream element is accepted and ERROR reports INVALID_DIMENSION

#### Scenario: Timeout
- **WHEN** a valid active job exceeds TIMEOUT_CYCLES before completion
- **THEN** BUSY clears, DONE stays clear, and ERROR reports TIMEOUT

#### Scenario: Invalid timeout configuration
- **WHEN** START is requested with TIMEOUT_CYCLES equal to zero
- **THEN** no stream element is accepted and ERROR reports INVALID_TIMEOUT
