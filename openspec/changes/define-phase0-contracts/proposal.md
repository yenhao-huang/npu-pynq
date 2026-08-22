## Why

The PE, systolic array, DMA integration, runtime, and exporter need one stable
definition of arithmetic and software-visible behavior. Without a Phase 0
contract, each Phase 1 component can pass local tests while disagreeing at
signed limits, rounding boundaries, tensor layout, or transaction control.

## What Changes

- Define a bit-accurate INT8 input, INT32 accumulation, and INT8
  requantization contract, including rounding, saturation, overflow, reset,
  and tensor layout behavior.
- Define hardware ABI version 1 for capability discovery, matrix job control,
  status/error reporting, stream ordering, alignment, and compatibility.
- Add a deterministic PYNQ-Z1 performance and resource model that separates
  ideal compute cycles, transport time, overhead, and measured acceptance.
- Implement executable Python golden-model, ABI, and performance-model modules
  with focused unit tests so later phases consume contracts rather than prose.

## Capabilities

### New Capabilities

- `numeric-contract`: Bit-accurate arithmetic, requantization, overflow, and
  tensor layout behavior shared by RTL, exporter, runtime, and verification.
- `hardware-abi`: Versioned control, capability, error, alignment, and matrix
  stream contract shared by the overlay and Python runtime.
- `performance-model`: Reproducible compute, bandwidth, resource, and
  acceptance calculations for the PYNQ-Z1 target.

### Modified Capabilities

None.

## Impact

- Adds contract implementations under `src/test/model/` and focused tests
  under `src/test/tests/`.
- Establishes interfaces that Phase 1A RTL and verification, Phase 1B Tcl and
  runtime, and Phase 1C examples must import or match.
- Does not add RTL, generated Vivado projects, bitstreams, board images, or
  external runtime dependencies beyond Python and NumPy.
- Related issue: `#2`; stacked repository foundation: draft PR `#11`.
