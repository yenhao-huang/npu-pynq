## Context

Since #60, `npu_matrix_controller` stores A in one synchronous bank per row and
B in one bank per column, and lets compute advance during B loading once each
B row is complete. A dense packed beat can span a row boundary (whenever K or
N is not a multiple of `IN_BYTES`), so a beat does not map onto one bank
write.

## Decisions

- **Dense stream, hardware realignment.** Software keeps sending dense byte
  buffers; the DMA packs them. A 2*`IN_BYTES`-byte row aligner sits between
  the stream and the banks and emits one row-aligned word per cycle: the next
  `IN_BYTES` bytes of the current row, or the rest of the row. Lanes past the
  end of a row carry the next row's bytes and land in bank locations that the
  compute schedule never marks valid, so no byte enables are needed. This is
  the realignment a DMA data-realignment engine performs, applied per row.
- **Bank writes.** An A bank takes the whole word in one write and is read a
  byte at a time, the write-wide/read-narrow asymmetric RAM pattern in UG901.
  B bank `c` takes lane `c % IN_BYTES` of word `c / IN_BYTES` of each B row,
  so every bank keeps one write port. The compute read path is #60's,
  unchanged.
- **Backpressure.** TREADY drops while the aligner cannot take a whole beat
  after this cycle's emission, and once a frame is fully accepted. A B row
  counts as resident only after its last word is written, which preserves
  #60's overlap invariant.
- **Strict beat checking.** Each beat must match the bytes the frame still
  owes: full TKEEP before the end, then TLAST and a low-lane TKEEP mask of the
  remainder. AXI DMA MM2S without DRE, reading from an aligned buffer,
  produces exactly this shape.

## Cost

A frame of R rows of L bytes loads in R*ceil(L/IN_BYTES) cycles, not
ceil(R*L/IN_BYTES). The two are equal when L is a multiple of `IN_BYTES`,
which holds for the full 8x8 tiles and for most ResNet-18 reductions; the
worst case, L = 1, is no slower than the byte-wide stream.

## Risks

- Timing and resource use at 100 MHz, and whether Vivado infers the
  asymmetric BRAM, must be confirmed by Vivado on the self-hosted runner.
