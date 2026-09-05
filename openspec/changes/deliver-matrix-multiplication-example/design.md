# Design: Logical matrix multiplication over the Phase 1B runtime

## Context

`NPURuntime.run` submits exactly one physical MxK by KxN job and derives its
limits from HWH metadata. The Phase 1C example needs larger and edge-shaped
logical matrices without bypassing validation, DMA ordering, recovery, or ABI
negotiation.

## Goals

- Reuse `NPURuntime.run` for every physical tile.
- Produce deterministic hardware-requantized signed INT8 output matching a
  saturated NumPy INT64 reference for all accepted shapes.
- Enforce one finite logical-operation deadline across all tiles.
- Return measured elapsed time, tile count, operation count, and throughput.
- Keep the notebook free of direct MMIO, DMA sequencing, and production logic.

## Non-goals

- K tiling, output requantization, model execution, or ResNet operators.
- Buffer pooling or asynchronous overlap of DMA and compute.
- Treating host measurements as physical-board performance evidence.

## Decisions

### Tile M and N, not K

The multiplier partitions M by `runtime.max_m` and N by `runtime.max_n`. Each
tile retains the complete K dimension and is copied into dense C-contiguous
INT8 arrays before calling `runtime.run`.

K MUST be positive and no larger than `runtime.max_k`. The ABI saturates after
each MAC; independently computing K slices from zero and adding their partial
results is not generally equivalent to one ordered saturated accumulation.
Exact K tiling therefore needs a future hardware or ABI contract and is not
silently approximated here.

### Logical deadline

The wrapper computes one monotonic deadline. Before every physical submission
it passes only the remaining time to `NPURuntime.run`. It fails before starting
another tile when no time remains, so a large logical matrix cannot multiply
the configured timeout by its tile count.

### Result and metrics

`MatrixMultiplicationResult` contains an owned C-contiguous INT32 result and
immutable metrics: logical M/N/K, tile count, elapsed seconds, integer MAC
count, integer operation count (`2*M*N*K`), and operations per second. Zero
elapsed time is permitted under deterministic fake clocks and reports infinite
throughput rather than dividing by zero.

### Notebook boundary

The notebook loads a same-basename overlay through `load_pynq_runtime`, creates
the logical multiplier, runs normal and non-tile-aligned matrices, checks a
NumPy reference, repeats a job, and prints non-secret performance/provenance
evidence. It contains no saved outputs and no ad hoc MMIO or DMA operations.

## Risks

- Per-tile allocation overhead limits performance. Phase 1C measures it; buffer
  pooling is deferred until board evidence identifies it as material.
- A board outage blocks the final acceptance criterion. Host tests remain
  useful but are recorded as lower-layer evidence only.
