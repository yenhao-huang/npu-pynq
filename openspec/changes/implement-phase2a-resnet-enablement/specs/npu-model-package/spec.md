## Purpose

Defines a deterministic, versioned, integrity-checked package that transfers a
validated quantized graph from the exporter to the Phase 2 model runtime.

## ADDED Requirements

### Requirement: Versioned package contents

An exported package SHALL contain a canonical UTF-8 manifest and one packed
binary weight payload. The manifest SHALL identify package magic, format major
and minor versions, graph inputs and outputs, ordered commands, tensor
metadata, quantization parameters, memory plan, accumulator-safety proofs,
required ABI major and capability bits, payload byte count, and SHA-256 digest.
The package SHALL contain no absolute host paths or executable code.

#### Scenario: Unsupported package major
- **WHEN** runtime package major differs from the supported major
- **THEN** preflight rejects the package before allocating memory or accessing hardware

### Requirement: Deterministic serialization

Identical validated graph inputs and exporter version SHALL produce
byte-identical manifest and weight payload files regardless of mapping insertion
order, current directory, wall clock, or host path. Tensor and command
identifiers SHALL be stable and weights SHALL use an explicitly declared
little-endian packed order.

#### Scenario: Repeated export
- **WHEN** the same graph is exported twice in different directories
- **THEN** both manifest bytes, payload bytes, and SHA-256 digest are identical

### Requirement: Export validation

The exporter SHALL reject unsupported graphs, duplicate identifiers, invalid
shapes or quantization parameters, inconsistent command dependencies, unsafe
accumulator bounds, malformed weights, and integer size overflow before
publishing either final package file.

#### Scenario: Failed export is not partially published
- **WHEN** validation fails after export begins
- **THEN** neither final manifest nor final weight payload is replaced by partial output

### Requirement: Package integrity and bounds

The runtime SHALL verify manifest structure, exact payload length, SHA-256
digest, every referenced payload range, tensor range, command reference, memory
range, required ABI, and capability bits before the first model operation.
Unknown commands or references and overlapping packed-weight ranges SHALL be
rejected explicitly.

#### Scenario: Corrupted weight byte
- **WHEN** one packed payload byte differs from the manifest digest
- **THEN** package preflight fails before physical runtime submission
