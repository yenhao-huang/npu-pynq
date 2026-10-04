## Why

The operand input stream is one byte wide end to end: AXI DMA MM2S sends 8-bit
beats and the controller writes one INT8 element per handshake. Loading A for a
16 x 256 tile takes 4096 cycles before compute starts, and the `M*K + K*N` load
term dominates every physical job (the #63 benchmark measured 1169 controller
cycles for an 8x8x64 job against 78 modeled compute cycles). The S_AXI_HP0 port
the DMA already uses is 64 bits wide, so the input path uses 1/8 of it. Issue
#93 records the problem and the prior art: AMD PG021 and UG585, the AMD
DPUCZDX8G, Gemmini (DAC 2021) and Google TPU v1 (ISCA 2017) all feed their
arrays through interfaces as wide as a row of operands, not one element.

## What Changes

- **BREAKING (AXI interface)**: the accelerator input stream carries
  `IN_BYTES` (default 8) consecutive operand bytes per beat on a
  `8*IN_BYTES`-bit TDATA, with TKEEP. The A and B frames keep their row-major
  byte order; a frame whose length is not a multiple of `IN_BYTES` ends with
  one partial beat whose TKEEP marks the remaining low lanes, exactly as AXI
  DMA MM2S emits an unaligned-length transfer.
- The controller validates TLAST and TKEEP per beat and reports any mismatch as
  `STREAM_LENGTH`, as it already does for TLAST.
- The overlay configures AXI DMA MM2S with a 64-bit memory-map and stream
  width, and sets `IN_BYTES = 8` on the accelerator. The HWH verifier requires
  both and the `s_axis_tkeep` port.
- The AXI-Lite register map, ABI identity, the S2MM output stream, the runtime,
  the export path and the numeric contract are unchanged. Software still sends
  `M*K` and `K*N` byte transfers from 64-byte-aligned buffers.

## Capabilities

### Modified Capabilities

- `matrix-accelerator-interface`: the two-frame input stream is packed
  `IN_BYTES` elements per beat with TKEEP on the final beat.
- `pynq-overlay-build`: the MM2S stream and memory map are 64 bits.

## Impact

Stall-free physical job length drops from
`M*K + K*N + M*N + K + M + N + 1` cycles to
`ceil(M*K/8) + ceil(K*N/8) + M*N + K + M + N + 1`. A BIT/HWH pair built before
this change no longer verifies and must be rebuilt.
