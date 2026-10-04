# Goal: A/B ping-pong buffer (issue #64)

## Prompt

實做 ping-pong buffer

實驗: 跑 16 個矩陣乘法 job 的 testbench 看 cycles 數是不是有下降
應該不會下降很多，因為目前 load data 的時間很重，但先實作好

驗收條件
得到 16 個矩陣乘法 job 的實驗數據

## Implementation

Branch `npu/issue64-b`, merged with `dev` after issue #59 (PR #60). The
ping-pong buffer is built on top of #59's operand storage:

- **Storage (#59):** one synchronous-read BRAM per array edge (ROWS A banks,
  COLUMNS B banks). The wavefront advances while the same job's B rows arrive.
- **Ping-pong (#64):** each BRAM is `2*MAX_K` deep, and the address MSB selects
  the ping-pong half. The load engine writes the half at `load_ptr`. The exec
  engine reads the half at `exec_ptr`, so the next job's operands stream in
  while the current job computes and drains.
- **Exec start:** the exec engine starts once a job's A frame has landed, so it
  keeps #59's in-job B-row overlap.
- **Software interface:** STATUS bit 3 (ACCEPT) tells software a half is free.
  CAPABILITIES bit 5 advertises PIPELINED_JOBS.

A driver that waits for `!BUSY` keeps #59 behaviour, minus one cycle per job:
the exec engine has no idle `STATE_CLEAR` cycle between `LOAD_B` and `COMPUTE`.

## Experiment: 16 matrix-multiply jobs

Testbench: `src/hw/tb/npu_matrix/tb_npu_matrix_controller_16jobs.sv`.
It runs 16 stall-free jobs (random INT8 operands, every output checked against
an exact INT32 golden model). The input stream is one byte per cycle and the
output is one INT32 per cycle with no backpressure. `span` is the wall-clock
length of all 16 jobs. `steady` is the mean `CYCLES` of jobs 1..15.

Three controllers, same testbench:

- **pre-#59**: `dev` at 667e072, serialized (`-DNPU_SERIAL_BASELINE`).
- **#59**: current `dev` (49dc74a), serialized (`-DNPU_SERIAL_BASELINE`).
  This is the hardware this change replaces.
- **#59 + ping-pong**: this branch, run both serialized and pipelined
  (START on `ACCEPT`).

| shape (M x N x K) | pre-#59 span | #59 span | #59+pp serial | **#59+pp pipelined** | saved vs #59 | steady/job #59 -> pipelined | speedup vs #59 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4x4x8    |  1616 |  1504 |  1488 |  **1068** | 29.0 % |   92 -> 65   | 1.41x |
| 4x4x64   |  9680 |  8672 |  8656 |  **8236** |  5.0 % |  540 -> 513  | 1.05x |
| 4x4x256  | 37328 | 33248 | 33232 | **32812** |  1.3 % | 2076 -> 2049 | 1.01x |
| 8x8x8    |  3536 |  3424 |  3408 |  **2148** | 37.3 % |  212 -> 129  | 1.59x |
| **8x8x64** (default) | 18768 | 17760 | 17744 | **16484** | **7.2 %** | **1108 -> 1025** | **1.08x** |
| 8x8x256  | 70992 | 66912 | 66896 | **65636** |  1.9 % | 4180 -> 4097 | 1.02x |
| 16x16x64 | 38480 | 37472 | 37456 | **33076** | 11.7 % | 2340 -> 2049 | 1.13x |

At 8x8x64, 16 jobs drop from 17760 cycles on current `dev` to 16484 cycles.
Against pre-#59 hardware (18768 cycles), #59 and ping-pong together save 12.2 %.

### Reading the numbers

As expected, the drop is small at realistic K. Every pipelined job settles at
exactly `M*K + K*N + 1` cycles, which is the time to stream the operands over
the 8-bit `s_axis` (8x8x64: 1025). The two overlaps hide different parts:

- #59 already hides most of the compute (K-1 steps) behind the job's own B frame.
- Ping-pong hides what is left (the `M + N` wavefront tail and the `M*N` output
  drain) behind the next job's load.

So the saving per job is roughly `M + N + M*N`, against a load of
`M*K + K*N` cycles. It is large when the output is large relative to the load
(8x8x8: 37 %, 16x16x64: 12 %) and close to nothing at K = 256. Further speedup
needs a wider or faster load path (for example a 32/64-bit `s_axis`, or reusing
B across row tiles). More banking cannot provide it.

### Reproduce

```bash
make -C src/test build/npu_matrix_controller_16jobs.ok   # default 8x8x64
# other shapes
iverilog -g2012 -s tb_npu_matrix_controller_16jobs \
  -Ptb_npu_matrix_controller_16jobs.ROWS=16 -Ptb_npu_matrix_controller_16jobs.COLUMNS=16 \
  -Ptb_npu_matrix_controller_16jobs.JOB_K=64 -o /tmp/16jobs.vvp \
  src/hw/rtl/npu_matrix/npu_matrix_controller.sv src/hw/rtl/systolic_array/*.sv \
  src/hw/tb/npu_matrix/tb_npu_matrix_controller_16jobs.sv && vvp /tmp/16jobs.vvp
# baselines: same command with -DNPU_SERIAL_BASELINE and the three RTL files
# taken from `git show <commit>:<path>` (667e072 for pre-#59, 49dc74a for #59)
```

### Validation and gaps

- `make -C src/test lint sim` passes: all 12 testbenches. These include:
  - the 16-job bench;
  - #59's `tb_npu_matrix_scaling` (its exact-latency formula is now one cycle
    lower);
  - the unchanged 34-case 8x8 controller regression.
- `tb_npu_matrix_scaling` also passes at SIZE = 4 and 16.
- Not yet run: Vivado synthesis, timing, and BRAM/LUT usage. Board validation
  is also outstanding. The doubled-depth memories should still map to one
  BRAM18 each, but this is unverified.
- `src/runtime/` still issues one job at a time. Software sees this speedup only
  after the runtime stages the next job on `ACCEPT`.
