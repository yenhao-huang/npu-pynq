## Purpose

Defines a host-testable PYNQ runtime that safely discovers, configures, executes,
and recovers bounded physical matrix jobs through MMIO and AXI DMA.

## ADDED Requirements

### Requirement: Metadata and ABI discovery
The runtime SHALL load a requested bitstream with its same-basename HWH, require
the expected accelerator and DMA instances in overlay metadata, derive MMIO
addresses and implementation limits from metadata rather than hard-coded host
addresses, and negotiate ABI magic, major/minor version, and required
capabilities before allocating or starting a job.

#### Scenario: Missing accelerator metadata
- **WHEN** the loaded HWH does not expose the expected accelerator instance or required parameters
- **THEN** runtime construction fails before any MMIO write or DMA transfer

#### Scenario: Incompatible ABI
- **WHEN** hardware reports the wrong magic, unsupported major version, or missing required capability
- **THEN** the runtime rejects the overlay before submitting a job

### Requirement: Matrix and buffer preflight
The runtime SHALL accept signed INT8 A and B matrices with compatible dense
two-dimensional shapes within the discovered physical M, N, and K limits. It
SHALL allocate separate contiguous A, B, and C buffers, verify 64-byte-aligned
non-wrapping physical ranges and sufficient sizes, and ensure C does not overlap
either input. C SHALL hold signed INT32 row-major values. Invalid dtype, rank,
shape, limits, allocation, alignment, range, or aliasing SHALL fail before
hardware configuration.

#### Scenario: Valid bounded matrices
- **WHEN** A is signed INT8 MxK and B is signed INT8 KxN within the discovered limits
- **THEN** preflight returns dense ABI register values and three valid DMA buffers

#### Scenario: Aliased output allocation
- **WHEN** the writable C physical range overlaps A or B
- **THEN** preflight rejects the job before configuring DMA or asserting START

### Requirement: Deterministic DMA and MMIO sequence
For each job, the runtime SHALL clear stale hardware state, program dense
dimensions, strides, and a finite hardware timeout, arm S2MM for exactly
4*M*N bytes, assert START, send the A frame, wait for its MM2S completion, send
the B frame, and then wait for hardware and S2MM completion. It SHALL flush
input buffers before DMA and invalidate output buffers before reading results.
Every poll or DMA wait SHALL have a finite monotonic software deadline.

#### Scenario: Successful transfer order
- **WHEN** all DMA and accelerator handshakes complete before their deadlines
- **THEN** the runtime returns a signed INT32 MxN result and the hardware status is DONE without ERROR

#### Scenario: Software timeout
- **WHEN** DMA or hardware completion does not occur before the software deadline
- **THEN** the runtime issues accelerator SOFT_RESET, attempts bounded DMA recovery, and raises TimeoutError without hanging the Python process

### Requirement: Exact hardware error propagation
After completion or abnormal termination, the runtime SHALL read STATUS and
ERROR and map every ABI v1 error code to a typed exception carrying the numeric
code and diagnostic context. It SHALL never return a result when ERROR is set,
DONE is clear, DMA lengths disagree, or TLAST/stream status indicates a failed
transaction.

#### Scenario: Stream length error
- **WHEN** hardware reports STREAM_LENGTH after malformed input
- **THEN** the runtime raises an error identifying STREAM_LENGTH and returns no matrix

### Requirement: Host-testable dependency boundary
The runtime SHALL isolate PYNQ-specific imports and accept injected overlay,
MMIO, DMA-channel, allocator, clock, and buffer doubles so ABI negotiation,
validation, sequencing, signed conversion, timeouts, and recovery can be tested
on a non-PYNQ host without pretending that those tests prove board behavior.

#### Scenario: Host fake-DMA test
- **WHEN** host tests provide deterministic fake MMIO, DMA, buffers, and time
- **THEN** the same public runtime sequence is observable without importing the PYNQ package
