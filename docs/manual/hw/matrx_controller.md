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

Operands are stored as one synchronous-read memory per array edge, the layout
introduced by issue #59: `gen_a_banks[r]` holds row `r` of `A`, and
`gen_b_banks[c]` holds column `c` of `B`. Each memory has one write port, fed
by `s_axis`, and one read port, feeding the array's boundary register. This
lets them map to block RAM.

Issue #64 makes each memory `2*MAX_K` deep. The address MSB selects the
ping-pong half:

```
A bank r : memory[{half, k}]  =  A[r][k]   of the job in that half
B bank c : memory[{half, k}]  =  B[k][c]   of the job in that half
```

Three one-bit pointers name the halves:

| Pointer | Owner | Advances on |
| --- | --- | --- |
| `alloc_ptr` | job admission | an accepted `START` |
| `load_ptr` | load engine, writes its half | the end of a B frame |
| `exec_ptr` | exec engine, reads its half | a retired job |

Because the pointers are independent, the load engine writes one half while the
exec engine reads the other. `occupancy` counts outstanding jobs and drives
`status_busy` (`occupancy != 0`) and `status_accept` (`occupancy != 2`).

The number of memories is unchanged: `ROWS` for A and `COLUMNS` for B, at
`2*MAX_K` bytes each, which is 512 bytes at `MAX_K = 256`. Doubling the depth
still fits one 18 Kb block RAM per memory, so the block RAM count need not
grow. This is an expectation only; section 10 notes that no Vivado run has
confirmed it.

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

**Exec engine** — owns the array and `m_axis`, and reads the half at
`exec_ptr`:

```
EXEC_IDLE --(entry's A frame landed)--> EXEC_CLEAR --> EXEC_COMPUTE --> EXEC_OUTPUT
EXEC_OUTPUT --(TLAST accepted)--> EXEC_IDLE, or straight to EXEC_CLEAR if the
                                  other entry's A frame has landed
```

`EXEC_CLEAR` zeroes the accumulators for one cycle. `EXEC_COMPUTE` walks
`K + M + N - 1` wavefront steps. While the same job's B frame is still loading,
a step is taken only if its B row is resident (`compute_step < load_outer`).
Otherwise the array, and every boundary register, holds. This is the in-job
overlap from issue #59. `EXEC_OUTPUT` emits `M*N` beats.

Two same-cycle handoffs keep a lone job free of bubbles:
- An accepted `START` puts an idle load engine straight into `LOAD_A`.
- The end of an A frame puts an idle exec engine straight into `EXEC_CLEAR`,
  which runs during the first B beat.

A lone job therefore costs one cycle less than under issue #59 alone. The
serialized controller spent an idle `STATE_CLEAR` cycle between `LOAD_B` and
`COMPUTE`, and the exec engine has no such cycle.

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

- For a single, isolated job this is the same busy-cycle count the pre-#64
  counter reported. The value itself is one cycle lower than under issue #59
  alone (section 5).
- For a burst, the per-job values sum to the burst's wall-clock length, so
  overlap shows up as smaller per-job numbers rather than as a separate metric.

On an error the counter holds the count accrued up to the abort and then stops.

## 9. Pipelining boundaries

What overlaps:

- **within a job** (issue #59): the wavefront advances as each B row lands;
- **across jobs** (issue #64): the next job's `LOAD_A` / `LOAD_B` with the
  current job's `EXEC_COMPUTE` and `EXEC_OUTPUT`, through the two halves;
- back-to-back jobs with no bubble between them, in either engine.

What still does not overlap:

- **more than two jobs.** The queue is two deep because there are two halves.
- **compute with the A frame.** The exec engine starts only once A is
  complete.
- **output with compute of the same job.** Results are read from the
  accumulators after the wavefront drains.
- **the load itself.** `s_axis` is 8 bits wide and accepts one element per
  cycle, so a job still costs `M*K + K*N` cycles of loading.

That last point sets the ceiling. For a lone job the cost is

```
1 + M*K + K*N   (start and load)  +  M + N   (wavefront tail)  +  M*N   (output)
```

At steady state the load engine never idles, so a pipelined job costs exactly
`M*K + K*N + 1` cycles. That holds whenever the previous job's tail and output
fit inside the next job's A frame. Ping-pong buffering hides the tail and the
output; it cannot hide the load. Going further needs a wider input stream, or
keeping an operand resident across tiles, not a third half.

## 10. Measured behaviour

`src/hw/tb/npu_matrix/tb_npu_matrix_controller_16jobs.sv` runs 16 stall-free
jobs twice: serialized (`START` waits for `!BUSY`), then pipelined (`START`
waits for `ACCEPT`). Every output is checked against an exact INT32 model. The
"#59 only" column runs the serialized pass against the controller on `dev`
before this change, with `-DNPU_SERIAL_BASELINE`.

| shape | #59 only, 16 jobs | this change, pipelined | saved | per job, steady |
| --- | ---: | ---: | ---: | --- |
| 8x8x8    |  3424 |  2148 | 37.3 % |  212 -> 129 |
| 8x8x64   | 17760 | 16484 |  7.2 % | 1108 -> 1025 |
| 8x8x256  | 66912 | 65636 |  1.9 % | 4180 -> 4097 |
| 16x16x64 | 37472 | 33076 | 11.7 % | 2340 -> 2049 |

`docs/goal/ping-pong-buffer-prompt.md` has the full sweep, including 4x4 and
the controller from before issue #59.

`tb_npu_matrix_controller_pipeline.sv` (2x2) and `_pipeline_8x8.sv` cover the
same mechanism with input stalls and output backpressure injected. They also
check:
- half swapping;
- `SOFT_RESET` with a load in flight;
- `BUSY_START` on a full queue;
- `STREAM_LENGTH` on a queued pipeline;
- a queued job's own timeout.

`tb_npu_matrix_scaling.sv` keeps the issue #59 checks at 4, 8 and 16:
- exact no-stall latency;
- stalled and partial tiles;
- B aborted after overlapped compute.

No Vivado run has been made for the doubled-depth memories. Block RAM, LUT
and timing numbers belong in `docs/exp/` once measured.
