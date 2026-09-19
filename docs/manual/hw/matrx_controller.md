# `npu_matrix_controller`

`src/hw/rtl/npu_matrix/npu_matrix_controller.sv` turns one AXI-Lite job
description plus one AXI-Stream operand frame into one AXI-Stream result frame.
It owns the operand storage and the schedule that walks `npu_systolic_array`;
it does not own the register decode, which is `npu_axi_lite_regs`.

This document was created by the change that added the A/B ping-pong buffer
(issue #64). It describes the controller as it stands after that change.

## 1. Parameters

| Parameter | Meaning |
| --- | --- |
| `ROWS` | systolic rows, and the maximum `M` |
| `COLUMNS` | systolic columns, and the maximum `N` |
| `MAX_K` | maximum reduction length `K` per job |

## 2. Interfaces

Unchanged by issue #64. AXI-Lite job configuration and status arrive as the
`cfg_*`, `start_pulse`, `soft_reset_pulse` and `status_*` ports;
operands arrive on an 8-bit `s_axis`; results leave on a 32-bit `m_axis`.
`status_accept` is the one added port, and it is carried on a previously
reserved bit of the existing `STATUS` register rather than a new address.

## 3. Job admission

A job is latched into a queue entry on an accepted `START`. The queue holds two
entries, one per operand bank.

| Condition at `START` | Outcome |
| --- | --- |
| a free entry, configuration valid | accepted, `STATUS.ACCEPT` may clear |
| both entries occupied | `ERR_BUSY_START`, outstanding jobs keep running |
| configuration invalid | `ERR_INVALID_DIMENSION` / `_STRIDE` / `_TIMEOUT`, pipeline quiesces |

`STATUS.ACCEPT` (bit 3) reports that an entry is free. It also gates the
configuration registers: `M`, `N`, `K`, the strides and `TIMEOUT_CYCLES` are
writable whenever `ACCEPT` is set, which is what lets software stage the next
job while the current one runs. Before issue #64 the same registers were gated
on `!BUSY`, which made staging impossible.

Each accepted job carries its own `M`, `N`, `K` and timeout. A queued job's
timeout runs from its own acceptance, so it covers the time the job spends
waiting for the array, and software must size it accordingly.

## 4. Datapath and operand banks

`a_buffer` and `b_buffer` each hold **two banks**:

```
a_buffer [0 : 2*ROWS*MAX_K - 1]      bank b occupies [b*ROWS*MAX_K    ...]
b_buffer [0 : 2*MAX_K*COLUMNS - 1]   bank b occupies [b*MAX_K*COLUMNS ...]
```

Three one-bit pointers name the banks:

| Pointer | Owner | Advances on |
| --- | --- | --- |
| `alloc_ptr` | job admission | an accepted `START` |
| `load_ptr` | load engine, writes its bank | the end of a B frame |
| `exec_ptr` | exec engine, reads its bank | a retired job |

Because the pointers are independent, the load engine writes one bank while the
exec engine reads the other. `occupancy` counts outstanding jobs and drives
`status_busy` (`occupancy != 0`) and `status_accept` (`occupancy != 2`).

Element layout inside a bank is unchanged: `A` is row-major with stride
`MAX_K`, `B` is row-major with stride `COLUMNS`, and the compute schedule reads
`a_buffer[base + row*MAX_K + k]` and `b_buffer[base + k*COLUMNS + column]`.

Doubling the banks doubles operand storage. For the 8x8 target with
`MAX_K = 256` that is 2 x 8 x 256 bytes for `A` and 2 x 256 x 8 bytes for `B`,
4 KiB each. See section 10 for the measured implementation cost.

## 5. State machines

The single serialized FSM was replaced by two that run concurrently.

**Load engine** — owns `s_axis` and the bank at `load_ptr`:

```
LOAD_IDLE --(queue entry waiting)--> LOAD_A --(A frame TLAST)--> LOAD_B
LOAD_B --(B frame TLAST)--> LOAD_IDLE, or straight to LOAD_A if another
                            entry is already waiting
```

`s_axis_tready` is asserted in `LOAD_A` and `LOAD_B` only. A beat whose `TLAST`
does not match the frame's last element raises `ERR_STREAM_LENGTH`.

**Exec engine** — owns the array and `m_axis`, and reads the bank at
`exec_ptr`:

```
EXEC_IDLE --(entry loaded)--> EXEC_CLEAR --> EXEC_COMPUTE --> EXEC_OUTPUT
EXEC_OUTPUT --(TLAST accepted)--> EXEC_IDLE, or straight to EXEC_CLEAR if the
                                  other bank is already loaded
```

`EXEC_COMPUTE` runs for `K + M + N - 1` steps, unchanged. `EXEC_OUTPUT` emits
`M*N` beats, unchanged.

Two same-cycle handoffs keep a lone job at its previous latency: an accepted
`START` puts an idle load engine straight into `LOAD_A`, and the end of a B
frame puts an idle exec engine straight into `EXEC_CLEAR`. A job that arrives
alone therefore spends exactly as many cycles in the controller as it did
before issue #64.

## 6. Errors

| Code | Raised by | Effect on outstanding jobs |
| --- | --- | --- |
| 1, 2, 6 | invalid configuration at `START` | all dropped |
| 3 `BUSY_START` | `START` with both entries occupied | none, they keep running |
| 4 `STREAM_LENGTH` | a `TLAST` in the wrong place | all dropped |
| 5 `TIMEOUT` | any outstanding job exceeding its own timeout | all dropped |

`status_error` stays sticky until `SOFT_RESET`, or until a `START` accepted
while the queue is empty.

## 7. `SOFT_RESET`

`SOFT_RESET` drops both queue entries, both engines, the in-flight load frame,
every pointer and every counter, and clears `BUSY`, `DONE`, `ERROR` and
`CYCLES`. Bank contents are left as they are; a bank is only read for a job
whose load frame completed, so stale bytes are never observable.

## 8. The `cycles` counter

`CYCLES` reports the **issue interval** of the most recently retired job: the
number of cycles since the previous retirement, or since the engine went busy
for the first job in a burst.

- For a single, isolated job this is bit-identical to the pre-#64 counter,
  which is what keeps records comparable against the issue #63 baseline.
- For a burst, the per-job values sum to the burst's wall-clock length, so
  overlap shows up as smaller per-job numbers rather than as a separate metric.

On an error the counter holds the count accrued up to the abort and then stops.

## 9. Pipelining boundaries

What overlaps, after issue #64:

- the next job's `LOAD_A` / `LOAD_B` with the current job's `EXEC_COMPUTE` and
  `EXEC_OUTPUT`, through the A/B banks;
- back-to-back jobs with no bubble between them, in either engine.

What still does not overlap:

- **more than two jobs.** The queue is two deep because there are two banks.
- **within a single job.** `LOAD_A`, `LOAD_B`, `EXEC_COMPUTE` and `EXEC_OUTPUT`
  remain sequential for the job that owns them; `COMPUTE` does not begin on
  partially loaded operands.
- **output with compute of the same job.** Results are read from the
  accumulators after the wavefront drains.
- **the load itself.** `s_axis` is 8 bits wide and accepts one element per
  cycle, so a job still costs `M*K + K*N` cycles of loading.

That last point sets the ceiling. Per job the serialized cost is

```
M*K + K*N   (load)  +  1 + K + M + N - 1   (clear and compute)  +  M*N (output)
```

and the pipelined steady-state cost is the larger of the load term and the
compute-plus-output term. Because the byte-wide stream makes the load term
dominate at every shape this design targets, ping-pong buffering hides the
compute and output tail behind the next load, and cannot hide the load itself.
Going further requires a wider input stream, or keeping an operand resident
across tiles, not a third bank.

## 10. Measured behaviour

From `src/hw/tb/npu_matrix/tb_npu_matrix_controller_pipeline_8x8.sv`, which
runs stall-free 8x8x64 physical jobs, first serialized and then pipelined:

| Measurement | Cycles |
| --- | --- |
| single 8x8x64 job, serialized | 1171 |
| three jobs, serialized | 3519 |
| three jobs, pipelined | 3223 |
| saved | 296 |

The single-job figure reproduces the 1169-cycle baseline recorded by issue #63,
which is what makes the two columns comparable. The saving is 148 cycles for
each job that has a predecessor to hide behind — `1 + K + M + N - 1 + M*N`
with `K = 64`, `M = N = 8` — so steady-state cost per job falls from 1173 to
1025 cycles, a 12.6 % reduction, approaching the 1024-cycle load bound.

`tb_npu_matrix_controller_pipeline.sv` covers the same mechanism at 2x2 with
input stalls and output backpressure injected, and additionally checks bank
swapping, `SOFT_RESET` with a load in flight, `BUSY_START` on a full queue,
`STREAM_LENGTH` on a queued pipeline, and a queued job's own timeout.

Synthesis and timing numbers for the doubled banks are recorded in
`docs/exp/`.
