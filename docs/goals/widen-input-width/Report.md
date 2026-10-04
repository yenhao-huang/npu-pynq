# Widen the operand input width: simulation report

The operand input stream used to carry one INT8 element per cycle. Since #94
it carries eight per 64-bit beat. In RTL simulation, a stall-free 8 x 8 job
with K = 256 drops from 4,178 cycles to 596 (7.0x). A 16 x 16 job with K = 256
drops from 8,482 to 1,316 (6.4x).

| | |
| --- | --- |
| Issue | [#93](https://github.com/yenhao-huang/npu-pynq/issues/93) |
| Pull request | [#94](https://github.com/yenhao-huang/npu-pynq/pull/94), merged into `dev` as `3845645` |
| OpenSpec change | [`openspec/changes/widen-operand-input-stream/`](../../../openspec/changes/widen-operand-input-stream/proposal.md) |
| Design | [design.md](design.md) |
| Reproduce | [reproduce.md](reproduce.md) |
| Raw output | [metrics.json](metrics.json) |
| Scope | RTL simulation at an assumed 100 MHz clock, excluding host setup, DMA setup and DDR stalls. This is not Vivado timing and not a board measurement. |

## Problem

The whole input path was one byte wide: AXI DMA MM2S was configured with an
8-bit stream, and the controller wrote one byte per handshake. Loading A for a
16 x 256 tile took 4,096 cycles before compute could start. The S_AXI_HP0 port
that the DMA already uses is 64 bits wide, so the input path used only 1/8 of
it.

## Method

The method is to widen the stream to the native 64-bit HP port width and pack
eight INT8 operands per beat. Packing is done by the DMA; software still sends
dense byte buffers. A 16-byte row aligner then turns the packed stream into
row-aligned words for the per-row A banks and per-column B banks that #60
introduced. [design.md](design.md) walks through it with a worked example.

Prior art for feeding an array through an interface as wide as a row of
operands, rather than one element at a time:

- AMD PG021, AXI DMA: the MM2S stream width is configurable from 8 to 1,024
  bits.
- AMD UG585, Zynq-7000 TRM: the S_AXI_HP ports are 32 or 64 bits wide and are
  meant for PL accelerators.
- AMD DPUCZDX8G (PG338): uses wide AXI masters for data.
- Gemmini (Genc et al., DAC 2021): writes one scratchpad row per cycle.
- Google TPU v1 (Jouppi et al., ISCA 2017): uses a 256-byte-wide Unified
  Buffer.

## Cycle model

Stall-free job length:

| Controller | Cycles |
| --- | --- |
| Byte-wide stream, with #60's B/compute overlap | `M*K + K*N + M*N + M + N + 2` |
| Packed stream, this change | `M*ceil(K/8) + K*ceil(N/8) + M*N + M + N + 4` |

- **Load terms:** each frame takes one cycle per row-aligned word, so A (M rows
  of K bytes) takes `M*ceil(K/8)` cycles and B (K rows of N bytes) takes
  `K*ceil(N/8)` cycles.
- **Extra 2 cycles:** the aligner spends one fill cycle at the start of each of
  the two frames.
- **Assertions:** `tb_npu_matrix_scaling` and `tb_npu_matrix_controller_8x8`
  check this formula exactly, the latter for 34 shapes. Every stall-free row
  below matches it.

## Results

Simulation of `tb_npu_matrix_scaling` with square full tiles and no stalls.
The baseline and #60 columns come from
[`docs/exp/2026-09-11-hardware-scaling.md`](../../exp/2026-09-11-hardware-scaling.md).

| Array | K | v1.0.6 baseline | #60 (banks + overlap) | This change | vs #60 | vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4x4 | 1 | 34 | 34 | 33 | 1.03x | 1.03x |
| 4x4 | 32 | 313 | 282 | 76 | 3.71x | 4.12x |
| 4x4 | 256 | 2,329 | 2,074 | 412 | 5.03x | 5.65x |
| 8x8 | 1 | 98 | 98 | 93 | 1.05x | 1.05x |
| 8x8 | 32 | 625 | 594 | 148 | 4.01x | 4.22x |
| 8x8 | 256 | 4,433 | 4,178 | 596 | 7.01x | 7.44x |
| 16x16 | 1 | 322 | 322 | 310 | 1.04x | 1.04x |
| 16x16 | 32 | 1,345 | 1,314 | 420 | 3.13x | 3.20x |
| 16x16 | 256 | 8,737 | 8,482 | 1,316 | 6.45x | 6.64x |

Throughput at the assumed 100 MHz clock, MACs / cycles. PE utilization is
MACs / (array size x cycles).

| Array | K | #60 GMAC/s | This change GMAC/s | PE utilization |
| --- | ---: | ---: | ---: | ---: |
| 4x4 | 256 | 0.1975 | 0.9942 | 62.14% |
| 8x8 | 256 | 0.3921 | 2.7490 | 42.95% |
| 16x16 | 256 | 0.7726 | 4.9799 | 19.45% |

### Reading the results

- **Large K gains the most.** Load is the dominant term there and shrinks by
  8x. What remains, the `M*N` output serialization and the drain, did not
  change, so the end-to-end gain is 5x to 7x rather than 8x.
- **K = 1 gains almost nothing.** Each A row is a single byte, which costs one
  word, the same as before. Output serialization (`M*N`) dominates these jobs.
- **16 x 16 is now output-bound.** At K = 256, 256 of its 1,316 cycles are
  result output on the 32-bit S2MM stream. Widening the output side is the next
  lever and is outside this change.
- **Phase columns in `metrics.json`.** The phase split there (`load`,
  `compute`, `output`) classifies cycles by `s_axis_tready`. Since the aligner
  can hold TREADY low while it still writes words, some load cycles are
  counted as `compute`. Use `cycles` for comparisons.

### Packing efficiency

A frame of R rows of L bytes arrives as `ceil(R*L/8)` beats, but the
aligner needs `R*ceil(L/8)` words, because each row is rounded up on its own.
Each row loses at most 7/8 of a word, so the excess is bounded:

```text
R*ceil(L/8) - ceil(R*L/8)  <=  R - ceil(R/8)  <  R
```

The bound is reached when `L mod 8 = 1`, where every row ends with one byte
that costs a whole word. For a whole job, A loses fewer than M cycles and B
fewer than K cycles against an ideal packed stream.

| R (rows) | L (row bytes) | Beats `ceil(R*L/8)` | Words `R*ceil(L/8)` | Excess |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 256 | 256 | 256 | 0 |
| 2 | 13 | 4 | 4 | 0 |
| 8 | 147 | 147 | 152 | 5 |
| 8 | 9 | 9 | 16 | 7 (the bound) |
| 8 | 1 | 1 | 8 | 7; same as the byte-wide stream, never slower |

There is no excess whenever L is a multiple of 8. That includes full 8x8 B
tiles (N = 8) and most ResNet-18 reductions. The first convolution,
K = 3 x 7 x 7 = 147, loses 5 cycles per 8-row A tile.

## Correctness

| Check | Result |
| --- | --- |
| `make -C src/test lint sim` | PASS. Verilator lint is clean and all 9 testbenches pass. |
| Controller error paths | PASS: early and missing TLAST (A and B), short and long final TKEEP, partial TKEEP mid-frame. |
| Partial final beats | PASS: K = 13 job, 26-byte frames ending in a 2-byte beat. |
| Random shapes | PASS: 34 8x8 cases with exact cycle count; 4/8/16 scaling with stalls, backpressure, and aborts during overlapped B load. |
| Mutation: TKEEP check removed | Detected; `tb_npu_matrix_controller` fails. |
| Mutation: aligner ignores buffer room | Detected; `tb_npu_matrix_scaling` reports wrong results. |
| `python3 -m unittest discover -s src/test/tests` | PASS: 145 tests, including rejection of a byte-wide overlay. |

## Resource estimate

Yosys `synth_xilinx -family xc7` on the 8x8 accelerator, comparing `dev`
before #94 with `dev` after it. This is an estimate, not Vivado.

| Resource | Before | After |
| --- | ---: | ---: |
| LUT | 8,402 | 9,223 (+9.8%) |
| FF | 4,675 | 4,872 |
| DSP48E1 | 64 | 66 |
| Block RAM | 16 x RAMB18 (about 8 tiles) | 8 x RAMB18 + 8 x RAMB36 (about 12 tiles) |

The A banks now need 64-bit write ports, which takes a RAMB36. Yosys did infer
the write-wide/read-narrow asymmetric RAM.

## Not yet verified

- **Vivado:** synthesis, routed timing at 100 MHz, and whether Vivado infers
  the asymmetric BRAM. The 8x8 overlay had 0.079 ns of setup slack before #60.
- **Board:** a PYNQ-Z1 run, including end-to-end ResNet-18 time.

Both run in the `release/v1.0.7` CD run
([37206961783](https://github.com/yenhao-huang/npu-pynq/actions/runs/37206961783)).
That run is waiting for the board, which was unreachable from the runner on
2026-10-04. Update this report with its numbers once it passes.
