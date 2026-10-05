## Purpose

Packs the matrix input stream so that one beat carries several INT8 operands,
cutting the operand load phase by the beat width.

## MODIFIED Requirements

### Requirement: Two-frame matrix input stream
After an accepted START, the accelerator SHALL consume exactly M*K signed INT8
A elements in row-major order as one AXI4-Stream frame, followed by exactly K*N
signed INT8 B elements in row-major order as a second frame. Each input beat
SHALL carry IN_BYTES consecutive elements of its frame on an 8*IN_BYTES-bit
TDATA, the earliest element in bits [7:0]. IN_BYTES SHALL be a power of two,
visible in generated HWH metadata, and 8 in the PYNQ-Z1 overlay. Every beat
before the final beat of a frame SHALL have all TKEEP bits set; the final beat
SHALL assert TLAST and set exactly the low TKEEP bits that cover the remaining
elements. TVALID data and phase state SHALL remain stable while TREADY is
deasserted.

#### Scenario: Correct A and B frames
- **WHEN** A and B each present the declared element count packed IN_BYTES per beat, with TLAST and the matching TKEEP on their final beats
- **THEN** the controller begins the matrix computation with the exact signed elements received

#### Scenario: Frame length not a multiple of IN_BYTES
- **WHEN** a frame of 26 elements arrives on an 8-byte stream
- **THEN** it is accepted as three full beats and one final beat with TKEEP 0x03 and TLAST

#### Scenario: Early or missing TLAST
- **WHEN** TLAST is accepted before the beat holding the declared final element or is absent on that beat
- **THEN** BUSY clears, DONE remains clear, ERROR reports STREAM_LENGTH, and no successful output frame is produced

#### Scenario: Wrong TKEEP
- **WHEN** a beat's TKEEP does not equal the lanes the frame still owes
- **THEN** BUSY clears, DONE remains clear, ERROR reports STREAM_LENGTH, and no successful output frame is produced
