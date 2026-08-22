# Matrix multiplication example

## ADDED Requirements

### Requirement: Public runtime-only notebook

The repository SHALL provide `examples/matrix-multiplication/matrix_multiplication.ipynb` as a clean,
output-free notebook that loads the overlay and performs multiplication through
the generic `src.runtime` API and the example-local matrix multiplier.

#### Scenario: No hardware bypass

- **WHEN** the notebook source is inspected
- **THEN** it contains no direct MMIO writes, DMA channel sequencing, or PYNQ
  allocation calls

### Requirement: Reference and boundary coverage

The notebook SHALL compare NPU output against a NumPy reference for a normal
case, a non-tile-aligned case, and repeated execution, failing on any mismatch.

#### Scenario: Non-tile-aligned demonstration

- **WHEN** the notebook is executed on a 2x2 physical array
- **THEN** at least one demonstrated logical output has M or N not divisible by
  two
- **AND** the result equals the NumPy reference exactly

### Requirement: Board evidence

The notebook SHALL print non-secret logical dimensions, tile count, elapsed
time, operation count, and throughput for a physical-board run.

#### Scenario: Board unavailable

- **WHEN** the PYNQ-Z1 cannot be reached
- **THEN** host tests may pass but board execution and measured performance
  remain explicitly blocked
