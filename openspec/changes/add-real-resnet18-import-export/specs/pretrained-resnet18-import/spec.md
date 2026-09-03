## Purpose

Defines trustworthy acquisition and deterministic conversion of one real
pretrained ResNet-18 into the repository's existing signed-INT8 NPU contract.

## ADDED Requirements

### Requirement: Pinned public model source

The system SHALL identify the selected pretrained ResNet-18 by provider,
architecture, weight revision, immutable HTTPS URL, expected byte length,
complete SHA-256 digest, and applicable license metadata. Download SHALL fail
closed on missing metadata, transport failure, size or digest mismatch,
redirect outside the approved host, or an existing destination.

#### Scenario: Downloaded checkpoint differs
- **WHEN** the downloaded checkpoint does not match the pinned byte length and SHA-256
- **THEN** no model-ready marker or converted package is published

### Requirement: Deterministic supported conversion

The converter SHALL accept only the pinned ResNet-18 state dictionary and
deterministically produce a validated batch-one `QuantizedGraph` using signed
INT8 NHWC activations, signed INT8 HWIO convolution weights, signed INT32
biases, folded inference BatchNorm, supported pooling, residual, and classifier
operators, and the existing Q1.31 requantization contract. It SHALL reject
missing, additional, malformed, unsupported, or non-finite source records
before publishing output.

#### Scenario: Identical conversion inputs
- **WHEN** the pinned checkpoint, conversion configuration, and calibration inputs are byte-identical
- **THEN** repeated conversion produces byte-identical `.npu.json` and `.npu.bin` files

#### Scenario: Source architecture differs
- **WHEN** a checkpoint omits or adds a tensor relative to the pinned ResNet-18 architecture
- **THEN** conversion fails with the offending source key before NPU package publication

### Requirement: Residual-compatible quantization

The converter SHALL use zero-centered signed activation quantization and SHALL
assign identical quantization to every identity or projection residual pair
and its add output. Requantization multipliers, shifts, folded biases, and
per-output-channel weights SHALL be derived deterministically and remain within
the Phase 0 numeric bounds.

#### Scenario: Residual branches cannot share a valid scale
- **WHEN** calibration or source parameters cannot produce one representable quantization contract for both residual branches and their output
- **THEN** conversion fails at the identified block instead of inserting floating-point or implicit rescaling

### Requirement: Real-model execution evidence

Validation SHALL use the converted pretrained weights and at least one dense
signed-INT8 input with logical shape `(1, 224, 224, 3)`. It SHALL compare the
stem, first residual stage, and final output against an independent integer
reference and record source, input, converted-package, and result digests.
Calibration or smoke inputs SHALL NOT be represented as ImageNet accuracy
evidence.

#### Scenario: First stage disagrees
- **WHEN** any captured stem or first-stage tensor differs from the integer reference
- **THEN** real-model validation fails with the first tensor and index mismatch and emits no pass evidence
