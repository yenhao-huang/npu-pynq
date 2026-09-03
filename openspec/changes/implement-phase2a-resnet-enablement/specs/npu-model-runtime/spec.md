## Purpose

Defines safe package preflight and repeatable batch-one execution across the
public Phase 1 matrix runtime and deterministic host-side ResNet operators.

## ADDED Requirements

### Requirement: Complete preflight before execution

The model runtime SHALL validate the complete package, graph input name, signed
INT8 input dtype and shape, memory capacity, operator parameters, accumulator
proofs, ABI requirements, and runtime physical limits before submitting the
first physical matrix job.

#### Scenario: Invalid model input shape
- **WHEN** the provided input shape differs from package tensor metadata
- **THEN** execution fails before any physical runtime call

### Requirement: Public runtime dispatch

Every convolution and fully connected physical matrix job SHALL be submitted
through the public Phase 1 runtime API. Model execution SHALL NOT perform direct
MMIO, DMA sequencing, overlay discovery, or register polling.

#### Scenario: Fake physical runtime
- **WHEN** a valid package executes against a conforming fake Phase 1 runtime
- **THEN** observed calls contain only bounded dense signed INT8 matrix tiles with finite remaining timeouts

### Requirement: Host operator execution

Residual add, ReLU, max pooling, global average pooling, and flatten SHALL
execute according to the quantized operator capability without changing tensor
metadata or using floating-point arithmetic.

#### Scenario: Residual block integration
- **WHEN** convolution output and an identity branch are consumed by residual add followed by ReLU
- **THEN** the block output equals the bit-accurate command-by-command reference

### Requirement: Owned repeatable result

Each successful invocation SHALL return an owned C-contiguous signed INT8 output
whose shape matches package output metadata. Repeated invocations with changed
inputs SHALL not expose stale tensor, scratch, partial-sum, or output data from a
previous call.

#### Scenario: Repeated input changes
- **WHEN** one loaded package executes twice with different valid inputs
- **THEN** both outputs independently equal their references

### Requirement: Execution accounting and recovery

The runtime SHALL report immutable elapsed time, command counts by operator,
physical matrix job count, MAC count, operation count, and physical cycle sum.
On a physical job failure or timeout it SHALL discard partial model output,
propagate diagnostic context naming the command and tile, and leave subsequent
invocations able to use the Phase 1 runtime recovery contract.

#### Scenario: Physical job fails mid-model
- **WHEN** a matrix job raises a hardware or timeout error
- **THEN** no successful model result is returned and the error identifies the failing command and tile
