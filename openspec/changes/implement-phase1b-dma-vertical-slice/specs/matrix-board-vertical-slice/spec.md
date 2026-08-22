## Purpose

Defines the physical PYNQ-Z1 evidence required to claim that one real matrix
job traverses DDR, DMA, programmable logic, and back to software correctly.

## ADDED Requirements

### Requirement: Provenance-checked board deployment
Board validation SHALL use the repository-defined PYNQ-Z1 target and a matched
`npu_matrix.bit`/`npu_matrix.hwh` pair from the current source commit. Before
execution, evidence SHALL record the source commit, Vivado version, target part,
artifact hashes, PYNQ version, discovered accelerator and DMA instances,
control addresses, and physical implementation limits. Credentials SHALL not
be recorded.

#### Scenario: Wrong artifact provenance
- **WHEN** artifact hashes, basenames, source commit, target part, or HWH metadata do not match the intended build
- **THEN** the smoke test stops before loading or starting the accelerator

### Requirement: End-to-end signed matrix transaction
The board smoke test SHALL execute at least one M=2 N=2 K=2 signed INT8 job
containing both -128 and 127 through real PYNQ buffers, AXI DMA, accelerator
MMIO, and programmable logic. It SHALL compare every signed INT32 output with
the Phase 0 golden model and require DONE without ERROR, completed DMA channels,
a nonzero stable cycle count, and correct row-major output.

#### Scenario: Signed endpoint matrix passes
- **WHEN** the matching overlay runs A=[[-128,127],[7,-3]] and B=[[-1,2],[4,-5]]
- **THEN** C equals [[636,-891],[-19,29]] and all hardware and DMA completion assertions pass

### Requirement: Objective PASS and failure output
The smoke command SHALL print exactly `PASS: NPU DMA matrix vertical slice`
only after all provenance, metadata, DMA, status, cycle, and numeric assertions
succeed. Any mismatch, timeout, missing IP, overlay error, DMA error, or ABI
error SHALL exit nonzero with the failing layer identified. Simulation, host
tests, upload success, HWH inspection, or bitstream existence SHALL not
substitute for this physical-board PASS.

#### Scenario: Numeric mismatch
- **WHEN** any returned element differs from the golden result
- **THEN** the command exits nonzero, identifies the matrix coordinate and expected/actual values, and does not print the PASS marker
