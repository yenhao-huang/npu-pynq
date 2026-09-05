## Purpose

Defines how source-controlled Tcl reproducibly creates and audits the PYNQ-Z1
processing-system, DMA, accelerator, and deployable overlay implementation.

## ADDED Requirements

### Requirement: Reproducible PYNQ-Z1 block design
One repository-relative batch Tcl entry point SHALL recreate a project for part
xc7z020clg400-1 from a clean output directory, add only source-controlled RTL
and constraints, and build a block design containing Zynq PS7, AXI DMA in
simple mode, the matrix accelerator, AXI interconnect or SmartConnect, reset
logic, and interrupt concatenation. PS DDR and FIXED_IO SHALL be external. The
accelerator and all DMA interfaces SHALL use the same PS-generated clock and a
synchronous peripheral reset derived from it.

#### Scenario: Clean batch regeneration
- **WHEN** Vivado runs the Tcl from a clean checkout with the required license and board-independent part files
- **THEN** the project, block design, wrapper, synthesis, implementation, and bitstream are regenerated without GUI state

### Requirement: Fixed control and DDR connectivity
PS M_AXI_GP0 SHALL control the accelerator AXI4-Lite window at 0x43C00000 with
a 64-KiB range and the AXI DMA control window at 0x40400000 with a 64-KiB
range. DMA MM2S and S2MM memory masters SHALL reach PS DDR through S_AXI_HP0.
The MM2S stream SHALL be 8 bits, the S2MM stream SHALL be 8 bits, scatter-gather
SHALL be disabled, and both DMA interrupts SHALL reach IRQ_F2P through an
explicit concatenation block.

#### Scenario: Generated address metadata
- **WHEN** the implemented design emits HWH metadata
- **THEN** it identifies the accelerator and DMA instances, their exact control ranges, AXI protocols, stream widths, clock, reset, and interrupt connectivity

### Requirement: Auditable implementation evidence
The batch flow SHALL emit utilization, routed timing summary, DRC, route status,
address map, and build-log evidence under ignored output paths. A successful
build SHALL require synthesis and implementation completion, zero DRC errors,
no unrouted or partially routed nets, nonnegative routed WNS, zero setup-failing
endpoints, and successful write_bitstream completion. The command SHALL return
nonzero if any required gate fails.

#### Scenario: Timing failure
- **WHEN** routed WNS is negative or setup-failing endpoints are nonzero
- **THEN** the batch command fails and does not report the overlay accepted

### Requirement: Matched overlay artifacts
The flow SHALL produce one bitstream and one HWH file with basename
`npu_matrix` from the same current build, outside Git. Evidence SHALL record
their paths, hashes, timestamps, source commit, Vivado version, target part,
and accelerator/DMA address metadata. Generated projects, reports, bitstreams,
HWH files, journals, logs, checkpoints, and caches SHALL remain untracked.

#### Scenario: Stale HWH pair
- **WHEN** bitstream and HWH provenance or basename does not match
- **THEN** deployment validation rejects the pair before programming the FPGA
