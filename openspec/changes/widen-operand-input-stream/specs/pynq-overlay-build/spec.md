## Purpose

Widens the DMA operand path to the native 64-bit S_AXI_HP0 width.

## MODIFIED Requirements

### Requirement: Fixed control and DDR connectivity
PS M_AXI_GP0 SHALL control the accelerator AXI4-Lite window at 0x43C00000 with
a 64-KiB range and the AXI DMA control window at 0x40400000 with a 64-KiB
range. DMA MM2S and S2MM memory masters SHALL reach PS DDR through S_AXI_HP0.
The MM2S memory map and stream SHALL be 64 bits and drive the accelerator's
TKEEP, the accelerator SHALL be built with IN_BYTES equal to 8, the S2MM stream
SHALL be 32 bits, scatter-gather SHALL be disabled, and both DMA interrupts
SHALL reach IRQ_F2P through an explicit concatenation block.

#### Scenario: Generated address metadata
- **WHEN** the implemented design emits HWH metadata
- **THEN** it identifies the accelerator and DMA instances, their exact control ranges, AXI protocols, stream widths, IN_BYTES, clock, reset, and interrupt connectivity
