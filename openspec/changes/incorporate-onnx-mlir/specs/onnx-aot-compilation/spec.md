# ONNX AOT Compilation

## Purpose

Provide inspectable ONNX-to-MLIR compilation and preserve the CPU/NPU execution
contract while removing production ONNX Runtime operator sessions.

## ADDED Requirements

### Requirement: Inspect source graph compatibility

The exporter SHALL report source identity, imported opsets, operator domains,
attributes, tensor types and shapes, and unsupported operations before target
code generation. A successful inventory SHALL NOT imply successful compilation.

#### Scenario: Custom operator requires conversion

- **WHEN** the source contains an operator without a verified lowering
- **THEN** diagnostics identify its domain, name, opset, and required conversion
- **AND** production compilation does not silently execute it through ORT

### Requirement: Import before partition

The compiler SHALL import ONNX into inspectable MLIR before CPU/NPU partition.
An import failure SHALL preserve diagnostics and SHALL NOT produce a package
marked as successfully compiled.

#### Scenario: Supported fixture imports

- **WHEN** a supported fixture is imported with the selected toolchain
- **THEN** an ONNX dialect MLIR artifact is available before target lowering

### Requirement: Preserve existing NPU contracts

Compilation SHALL use the existing NPU partition input contract, ISA, driver,
and RTL without vendor changes. Partition SHALL happen before lowering erases
the contractions required for NPU selection.

#### Scenario: Quantized projection reaches partition

- **WHEN** a supported INT8 projection is compiled
- **THEN** the existing partition pass receives compatible tensor IR and weight metadata
- **AND** the package preserves the expected numerical and weight-layout contracts

### Requirement: Execute without production ORT sessions

The final compiled SmolLM2 package SHALL execute CPU operations and dispatch NPU
programs without ORT session models, the ORT shared library, or a manually built
ORT bridge object. It SHALL preserve prefill/decode entry points and KV semantics.

#### Scenario: Board executes compiled package

- **WHEN** the package runs on PYNQ-Z1 without ORT installed
- **THEN** prefill and decode agree with the independent numerical reference
- **AND** compiled ARM control flow submits NPU work through the existing driver

### Requirement: Gate removal on validation

The migration SHALL retain required legacy consumers until replacement numerical
and physical-board validation passes. It SHALL distinguish import, compilation,
and board validation results.

#### Scenario: Import passes but board validation is unavailable

- **WHEN** only source import has been verified
- **THEN** the implementation reports later gates as incomplete
- **AND** the production bridge is not deleted
