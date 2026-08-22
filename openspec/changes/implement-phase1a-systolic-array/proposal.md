## Why

Phase 1 needs a reusable compute fabric whose signed arithmetic is proven
against the Phase 0 golden model before DMA, control, or board integration can
hide arithmetic defects. A parameterized PE and systolic array provide the
smallest hardware slice that establishes this trust boundary.

## What Changes

- Add a signed INT8 processing element with per-MAC INT32 saturation, explicit
  clear/reset semantics, operand forwarding, validity propagation, and global
  clock-enable backpressure.
- Compose the PE into a parameterized rectangular output-stationary systolic
  array with row/column edge streams and packed accumulator outputs.
- Add deterministic self-checking SystemVerilog tests for signed endpoints,
  saturation, reset, stall, skew scheduling, and non-tile-aligned logical work.
- Add deterministic randomized vector generation and differential comparison
  against the Phase 0 Python golden model.
- Extend repository simulation entry points so focused PE/array tests are
  discoverable without adding generated simulation artifacts.

## Capabilities

### New Capabilities

- `signed-processing-element`: Signed saturating MAC behavior, operand/valid
  forwarding, reset, clear, and global-stall behavior for one PE.
- `systolic-array`: Parameterized output-stationary PE composition, skewed edge
  scheduling, result visibility, and whole-array backpressure semantics.
- `bit-accurate-rtl-verification`: Reproducible deterministic and randomized
  RTL comparisons against the Phase 0 numeric contract.

### Modified Capabilities

None. This change implements but does not alter the Phase 0
`numeric-contract`, `hardware-abi`, or `performance-model` requirements.

## Impact

- Adds synthesizable RTL under `src/hw/rtl/systolic_array/` and matching
  testbenches under `src/hw/tb/systolic_array/`.
- Adds deterministic vector generation under `src/test/vectors/` and focused
  Python verification under `src/test/tests/`.
- Updates `src/test/Makefile` only as needed for reliable module discovery and
  explicit top selection.
- Does not add AXI, DMA, MMIO, Tcl, constraints, Vivado projects, bitstreams,
  runtime code, or board claims; those remain Phase 1B / Issue `#4`.
- Related issue: `#3`; stacked dependency: Phase 0 draft PR `#12`.
