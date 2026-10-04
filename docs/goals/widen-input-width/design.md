# Packed operand stream: design

How eight INT8 operands per beat reach the per-row and per-column operand
banks. The decision record is in
[`openspec/changes/widen-operand-input-stream/design.md`](../../../openspec/changes/widen-operand-input-stream/design.md);
this page explains the mechanism.

## Data path

```text
DDR ── AXI DMA MM2S (64-bit) ──▶ s_axis_tdata[63:0], s_axis_tkeep[7:0], s_axis_tlast
                                    │  eight dense row-major INT8 bytes per beat;
                                    │  the final beat marks its valid bytes in TKEEP
                                    ▼
                         row aligner (16-byte buffer)
                                    │  one row-aligned 8-byte word per cycle
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
        A: one bank per row               B: one bank per column
        whole word in one write           bank c takes lane c % 8
        (write 64-bit, read 8-bit)        of word c / 8 of each B row
                      └─────────────┬─────────────┘
                                    ▼
             compute read path and systolic array: unchanged from #60
```

Software sends the same `M*K` and `K*N` byte transfers as before; the DMA does
the packing. The register map, runtime, export path and numeric contract are
unchanged.

## Why a row aligner is needed

#60 stores A as one bank per row. The stream is dense: the DMA cuts memory into
8-byte beats without regard to row boundaries. When K is not a multiple of 8, a
beat spans two rows.

Example: M = 2, K = 13. Row 0 is `a0..a12`, row 1 is `b0..b12`, 26 bytes in
total.

The banks must end up as:

```text
bank 0 (row 0): a0 a1 a2 ... a12
bank 1 (row 1): b0 b1 b2 ... b12
```

The DMA sends:

```text
beat 0: a0  a1  a2  a3  a4  a5  a6  a7
beat 1: a8  a9  a10 a11 a12 b0  b1  b2    <- two rows in one beat
beat 2: b3  b4  b5  b6  b7  b8  b9  b10
beat 3: b11 b12                           <- TKEEP = 0x03
```

Beat 1 cannot be written as it is:

- its first five bytes belong to bank 0 and its last three to bank 1;
- `b0` sits in lane 5 of the beat but must land at position 0 of bank 1.

## What the aligner does

The aligner is a queue. Beats enter at the back. Each cycle it removes from the
front the next piece of the current row: up to 8 bytes, or what is left of the
row. That piece belongs to exactly one row and starts at the right position, so
it is written straight into that row's bank.

| Cycle | Removed from the front (written) | Entered at the back | Left in the queue |
| --- | --- | --- | --- |
| 1 | nothing yet | beat 0 | a0–a7 |
| 2 | **a0–a7 → bank 0** | beat 1 | a8–a12, b0–b2 |
| 3 | **a8–a12 → bank 0** (only 5 left in row 0) | beat 2 | b0–b10 |
| 4 | **b0–b7 → bank 1** | beat 3 | b8–b12 |
| 5 | **b8–b12 → bank 1** | — | empty |

Each write goes to one bank only.

The lanes of a word past the end of a row carry the next row's bytes. They land
in bank positions at or beyond K, which the compute schedule never marks valid.
So no byte enables are needed.

## Backpressure

Normally 8 bytes enter and 8 bytes leave each cycle. At the end of a row fewer
leave (5 in cycle 3 above), so the queue grows. The queue holds 16 bytes. When
it cannot take a whole beat after this cycle's removal, `s_axis_tready` drops
and the DMA waits a cycle. TREADY also drops once a frame has been fully
accepted, so that B bytes never mix into A.

## B side

B is stored as one bank per column, written at address k, the B row. A B row
has N bytes, at most `COLUMNS`. The aligner hands out each B row as
`ceil(N/8)` words, and word w covers columns `8w .. 8w+7`. Bank c therefore
takes lane `c % 8` of word `c / 8`, one byte per bank per write.

#60's overlap rule is preserved: compute may read a B row only after the last
word of that row has been written.

## Stream checks

Each beat must match the bytes the frame still owes:

- every beat before the last has TKEEP all ones;
- the last beat has TLAST set and exactly the low TKEEP bits that cover the
  remaining bytes.

Anything else reports `STREAM_LENGTH`, as a TLAST mismatch already did.
