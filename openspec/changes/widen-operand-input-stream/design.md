## Context

`npu_matrix_controller` stored A at byte `r*MAX_K + k` and B at `k*COLUMNS + c`
and wrote one byte per handshake. A packed beat can straddle a row boundary
whenever K or N is not a multiple of `IN_BYTES`, so that layout would need
`IN_BYTES` independent write addresses per cycle. The 8x8 build met 100 MHz
with only 0.079 ns of setup slack, so the compute read path must not get
deeper.

## Decisions

- **Stream-order storage.** A and B are stored exactly as streamed, in
  `IN_BYTES`-byte words: A element `(r, k)` at byte `r*K + k`, B element
  `(k, c)` at byte `k*N + c`. Each accepted beat writes one word at a
  sequential address, so each buffer keeps a single write port.
- **No variable multiplier on the read path.** Per-row bases
  `a_row_base[r] = r*(K-1)` and per-column offsets `b_column_offset[c] =
  c*(N-1)` are registered when START is accepted, and `compute_step * N` is
  kept as a running sum. The A address is then one add
  (`a_row_base[r] + step`) and the B address one subtract
  (`step*N - b_column_offset[c]`), the same depth as the previous
  `step - r` index, followed by a word read and a lane select.
- **Strict beat checking.** Each beat must match the bytes the frame still
  owes: full TKEEP before the end, and TLAST plus a low-lane TKEEP mask of the
  remainder on the final beat. AXI DMA MM2S without DRE, reading from an
  aligned buffer, produces exactly this shape.
- **Transport only.** Software keeps sending byte buffers; packing happens in
  the DMA. Runtime, export and the numeric model are untouched.

## Risks

- Timing and resource use at 100 MHz must be confirmed by Vivado on the
  self-hosted runner; open-source simulation cannot show them.
