# `npu_matrix_controller`

`src/hw/rtl/npu_matrix/npu_accelerator/npu_matrix_core/controller/npu_matrix_controller.sv`
turns AXI-Lite job descriptions and packed AXI-Stream operand frames into
AXI-Stream result frames. It sequences the operand banks and the systolic array
in `npu_matrix_datapath`, but it stores no operands itself. Register decode
lives in `npu_axi_lite_regs`.

This document was created by the change that added the A/B ping-pong buffer
(issue #64). It describes the controller as it stands after that change. The
change is built on:
- the per-edge operand banks and in-job B-row overlap from issue #59;
- the packed 64-bit operand stream from issue #94;
- the module split from issue #62.

## 1. Parameters

| Parameter | Meaning |
| --- | --- |
| `ROWS` | systolic rows, and the maximum `M` |
| `COLUMNS` | systolic columns, and the maximum `N` |
| `MAX_K` | maximum reduction length `K` per job |
| `IN_BYTES` | INT8 operands per input beat (default 8, a power of two dividing `MAX_K`) |

## 2. Interfaces

**Interfaces unchanged by issue #64:**
- AXI-Lite job configuration and status arrive as the `cfg_*`, `start_pulse`,
  `soft_reset_pulse` and `status_*` ports.
- Operands arrive on an `8*IN_BYTES`-bit `s_axis` with `TKEEP`.
- Results leave on a 32-bit `m_axis`.

**Ports added by issue #64:**

| Port | Direction | Meaning |
| --- | --- | --- |
| `status_accept` | out | a queue entry is free; carried on `STATUS` bit 3 |
| `load_half` | out, to the datapath | the bank half the load engine writes |
| `exec_half` | out, to the datapath | the bank half the exec engine reads |

## 3. Job admission

A job is latched into a queue entry on an accepted `START`. The queue holds two
entries, one per bank half.

| Condition at `START` | Outcome |
| --- | --- |
| a free entry, configuration valid | accepted, `STATUS.ACCEPT` may clear |
| both entries occupied | `ERR_BUSY_START`, outstanding jobs keep running |
| configuration invalid | `ERR_INVALID_DIMENSION` / `_STRIDE` / `_TIMEOUT`, pipeline quiesces |

`STATUS.ACCEPT` (bit 3) reports that an entry is free. It also gates the
configuration registers: `M`, `N`, `K`, the strides and `TIMEOUT_CYCLES` are
writable whenever `ACCEPT` is set. This lets software stage the next job while
the current one runs. Before issue #64 the same registers were gated on
`!BUSY`, which made staging impossible. `CAPABILITIES` bit 5
(`PIPELINED_JOBS`) is advertised but not required.

Each accepted job carries its own `M`, `N`, `K` and timeout. A queued job's
timeout runs from its own acceptance, so it covers the time the job spends
waiting for the array. Software must size it accordingly.

## 4. Operand banks and halves

`npu_matrix_datapath` instantiates one `npu_operand_buffer` per array edge:
`gen_a_banks[r]` holds row `r` of `A`, and `gen_b_banks[c]` holds column `c`
of `B`. Each buffer has one write port, fed by the row aligner, and one
synchronous read port, feeding the array's boundary register.

Issue #64 makes each buffer `2 * 2**ceil(log2(MAX_K))` bytes deep. The address
MSB selects the half:

```
A bank r : write word {load_half, word}    read byte {exec_half, k}
B bank c : write byte {load_half, k}       read byte {exec_half, k}
```

Three one-bit pointers in the controller name the halves:

| Pointer | Owner | Advances on |
| --- | --- | --- |
| `alloc_ptr` | job admission | an accepted `START` |
| `load_ptr` (`load_half`) | load engine, writes its half | the end of a B frame |
| `exec_ptr` (`exec_half`) | exec engine, reads its half | a retired job |

Because the pointers are independent, the load engine writes one half while the
exec engine reads the other. `occupancy` counts outstanding jobs and drives
`status_busy` (`occupancy != 0`) and `status_accept` (`occupancy != 2`).

The number of buffers is unchanged: `ROWS + COLUMNS`. At `MAX_K = 256` each
holds 512 bytes, which still fits one 18 Kb block RAM. The block RAM count
should therefore not grow, but no Vivado run has confirmed this yet
(section 10).

## 5. State machines

**Load engine.** It owns `s_axis`, the row aligner and the half at `load_ptr`:

```
LOAD_IDLE --(entry waiting)--> LOAD_A --(A frame done)--> LOAD_B
LOAD_B --(B frame done)--> LOAD_IDLE, or straight to LOAD_A if the other
                           entry is already waiting
```

The row aligner, `load_remaining` and the per-beat `TLAST` / `TKEEP` check are
unchanged from issue #94. They now read the dimensions of the entry at
`load_ptr`. A beat whose `TLAST` or `TKEEP` does not match the bytes the frame
still owes raises `ERR_STREAM_LENGTH`.

**Exec engine.** It owns the array and `m_axis`, and reads the half at
`exec_ptr`:

```
EXEC_IDLE --(entry's A frame landed)--> EXEC_CLEAR --> EXEC_COMPUTE --> EXEC_OUTPUT
EXEC_OUTPUT --(TLAST accepted)--> EXEC_IDLE, or straight to EXEC_CLEAR if the
                                  other entry's A frame has landed
```

- `EXEC_CLEAR` zeroes the accumulators for one cycle.
- `EXEC_COMPUTE` walks `K + M + N - 1` wavefront steps. While the same job's B
  frame is still loading, a step is taken only if its B row is resident
  (`compute_step < align_row`). Otherwise the array and every boundary register
  hold. This is the in-job overlap from issue #59.
- `EXEC_OUTPUT` emits `M*N` beats.

Two same-cycle handoffs keep a lone job free of bubbles:
- An accepted `START` puts an idle load engine straight into `LOAD_A`.
- The end of an A frame puts an idle exec engine straight into `EXEC_CLEAR`.

A lone job therefore costs one cycle less than before issue #64. The serialized
controller spent an idle `STATE_CLEAR` cycle between `LOAD_B` and `COMPUTE`, and
the exec engine has no such cycle.

## 6. Errors

| Code | Raised by | Effect on outstanding jobs |
| --- | --- | --- |
| 1, 2, 6 | invalid configuration at `START` | all dropped |
| 3 `BUSY_START` | `START` with both entries occupied | none, they keep running |
| 4 `STREAM_LENGTH` | `TLAST` or `TKEEP` not matching the frame | all dropped |
| 5 `TIMEOUT` | any outstanding job exceeding its own timeout | all dropped |

`status_error` stays sticky until `SOFT_RESET`, or until a `START` accepted
while the queue is empty.

## 7. `SOFT_RESET`

`SOFT_RESET` drops:
- both queue entries and both engines;
- the in-flight frame and the aligner;
- every pointer and every counter.

It also clears `BUSY`, `DONE`, `ERROR` and `CYCLES`. Bank contents are left as
they are. A half is only read for a job whose A frame completed, and the
compute schedule masks every unwritten location, so stale bytes are never
observable.

## 8. The `cycles` counter

`CYCLES` reports the **issue interval** of the most recently retired job: the
number of cycles since the previous retirement, or since the engine went busy
for the first job in a burst.

- For a single, isolated job, this is the busy-cycle count the serialized
  counter reported, one cycle lower than before (section 5).
- For a burst, the per-job values sum to the burst's wall-clock length.
  Overlap therefore shows up as smaller per-job numbers, not as a separate
  metric.

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
- **output with the next job's compute.** Results are read straight from the
  array's accumulators, so the array cannot start the next wavefront until the
  `M*N` output beats have drained.

With the packed stream, the steady-state cost per job issued on `ACCEPT` is
whichever engine is slower:

```
load  = M*ceil(K/IN_BYTES) + K*ceil(N/IN_BYTES) + 2      (aligned words + fill)
exec  = 1 + (K + M + N - 1) + M*N                        (clear, wavefront, output)
cost  = load              if load >= exec
        exec + 1          otherwise
```

Before issue #94 the byte-wide load always dominated, so ping-pong could only
hide the compute tail and output. With 8 operands per beat, the array side is
the bound for mid-sized K: 8x8x64 and 16x16x64 are exec-bound. Further gains
need one of:
- output that does not stall the array, such as a result buffer or double
  accumulators;
- compute that starts before A completes.

## 10. Measured behaviour

`src/hw/tb/npu_matrix/tb_npu_matrix_controller_16jobs.sv` runs 16 stall-free
jobs twice, both through `npu_matrix_core` with 64-bit packed beats:
- serialized, where `START` waits for `!BUSY`;
- pipelined, where `START` waits for `ACCEPT`.

Every output is checked against an exact INT32 model. The "dev" column runs the
serialized pass against the core before this change, with
`-DNPU_SERIAL_BASELINE`.

| shape | dev, 16 jobs | this change, pipelined | saved | per job, steady |
| --- | ---: | ---: | ---: | --- |
| 8x8x8     |  1664 |  1438 | 13.6 % |  102 -> 89 |
| 8x8x64    |  3456 |  2390 | 30.8 % |  214 -> 145 |
| 8x8x256   |  9600 |  8309 | 13.4 % |  598 -> 514 |
| 16x16x64  |  8832 |  5846 | 33.8 % |  550 -> 353 |
| 16x16x256 | 21120 | 16709 | 20.9 % | 1318 -> 1026 |

`docs/goal/ping-pong-buffer-prompt.md` has the full sweep, including 4x4.

`tb_npu_matrix_controller_pipeline.sv` (2x2) and `_pipeline_8x8.sv` cover the
same mechanism with packed beats, and the 2x2 bench also injects input stalls
and output backpressure. Both check:
- half swapping;
- `SOFT_RESET` with a load in flight;
- `BUSY_START` on a full queue;
- `STREAM_LENGTH` on a queued pipeline;
- a queued job's own timeout.

`tb_npu_matrix_scaling.sv` keeps the issue #59 and #94 checks at 4, 8 and 16:
- exact no-stall latency;
- stalled and partial tiles;
- B aborted after overlapped compute.

No Vivado run has been made for the doubled-depth buffers. Block RAM, LUT
and timing numbers belong in `docs/exp/` once measured.
