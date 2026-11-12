# Kernel profiler plan

| Field | Value |
| --- | --- |
| Plan ID | 2026-1112-002 |
| Topic | Per-kernel time and bandwidth profiling for `pynqo-package/2` |
| Date | 2026-11-12 |
| Status | Draft |
| Based on | `origin/dev` at `95c35f7` |
| Related | #115 (per-stage timing, written for the deprecated JIT runtime), #157 (asynchronous delegates) |
| Issues | #172 (phase 1) |

## Goal

For one inference run (an LLM prefill/decode step, or a classifier forward) on
the PYNQ-Z1, report, **for every kernel**:

- wall time (total, count, min, max, share of the step);
- bytes moved and the resulting effective bandwidth (GB/s);
- operations and throughput (GOP/s);
- where it sits against the board's roofline (compute-bound or memory-bound,
  percentage of the attainable ceiling).

A *kernel* is, for now, **one ATen op** of the exported graph (a
`torch.aten.*` op in `torch.mlir`), the same granularity as the PyTorch
profiler. An ATen op runs either as CPU code in `model.so` or, for the
`_int_mm` contractions the NPU claims, as a `__delegate__` call. Each NPU
call is further split into its host-side phases and its PL time.

The output must answer "where does a 1.5 s decode token go?" without
guessing. Today's numbers (`docs/goals/sw_optimal/Results.md`) show 453 M PL
cycles (4.5 s at 100 MHz) across a 24-step run whose wall time is far larger,
and nothing attributes the rest.

## What exists today

| Layer | What is measured | Where |
| --- | --- | --- |
| Engine | prefill and decode wall time, tok/s | `src/runtime/tasks/llm/engine.py` (`generate`) |
| Package | `delegate_calls`, cumulative | `src/runtime/delegate/dispatch.c` |
| NPU driver | `calls`, `jobs`, `cycles` (sum of `ISA_CYCLES`), `macs`, `polls`, cumulative | `src/vendors/npu_pynq/driver/npu_rt/` |
| Hardware | `ISA_CYCLES`, `ISA_JOBS`, `ISA_INSTRUCTIONS` per program; 64-bit matrix `CYCLES` | `csr.sv`, `spec/abi.py` |

Gaps:

1. **No per-kernel attribution.** CPU kernels are inlined into one entry
   function in `model.so`; nothing marks where one op ends and the next begins.
   The emitted `torch.mlir` carries no debug locations either, so an ATen op
   cannot be recovered from the lowered linalg after the fact.
2. **NPU time is not split.** `npu_rt_execute` does `packA` (CPU memcpy), a
   per-segment `memcpy` of weights from the `weights.bin` mapping into the
   uncached `stage` buffer, `pynqRun` (start and poll), and `unpackC`. Only the
   PL cycles are kept, as a single total.
3. **No byte counts.** Neither software nor hardware records bytes moved, so
   bandwidth cannot be computed.
4. **No ceilings.** The board's achievable bandwidths have not been measured,
   so a GB/s number cannot be judged.

## Board ceilings (to be measured, phase 0)

| Path | Theoretical | Notes |
| --- | --- | --- |
| DDR3 (PS) | about 2.1 GB/s | 16-bit DDR3-1050 on the PYNQ-Z1; shared by the CPU and the PL |
| NPU `m_axi` to `S_AXI_HP0` | 800 MB/s per direction | 64-bit at `FCLK_CLK0` 100 MHz (`build_overlay.tcl`) |
| Cortex-A9 memcpy, cached | to measure | `packA`, `unpackC`, CPU kernels |
| Cortex-A9 memcpy into uncached `pynq.allocate` | to measure; expected to be low | the weight staging copy in `execute.cpp` |
| NPU INT8 MAC | `ROWS × COLUMNS × 100 M` MAC/s | from the overlay's `ROWS`/`COLUMNS` |
| Cortex-A9 NEON | to measure | 650 MHz on the PYNQ-Z1 (the Zynq-7020 is rated to 667 MHz); confirm on the board |

The measured values go into `src/vendors/npu_pynq/spec/board.py` as
`CEILINGS`, so reports and the roofline read one source.

## Design

Four levels, each optional and off by default. With profiling off, packages,
outputs and the numeric contract are unchanged, and the hot path does no extra
work.

```text
L0 engine        tokenize | prefill | decode step | sample | detokenize      (Python)
L1 aten op       aten.mul#12 | aten._int_mm#40 (npu) | aten.amax#41 ...    (model.so hooks)
L2 npu phases    packA | stage weights (per segment) | pl run | poll | unpackC  (libnpu_rt)
L3 pl counters   AXI read/write bytes, busy and stall cycles, array-active cycles  (RTL)
```

### L0: engine stages (Python only)

`ExportedLLM.generate` and the classifier engine take an optional `Profiler`
and record `tokenize`, `prefill`, each `decode`, `sample` and `detokenize`
with `time.perf_counter_ns`. The same profiler collects L1–L3 records for the
step it encloses, so every kernel record carries its step (prefill chunk or
decode position).

### L1: hooks around every ATen op (export side)

The hooks are placed where ATen identity is still exact: in `torch.mlir`, one
`torch.aten.*` op is one ATen op of the `ExportedProgram`. They are carried
through the lowering as opaque marker ops and become calls only once tensors
are buffers.

**1. Insert markers (pynqo-middle-end, torch dialect).** A new rewrite,
`--profile-aten`, runs first among our middle-end rewrites. For each
`torch.aten.*` op it assigns an id and threads the op's tensor operands and
results through two marker ops of a small `pynqo_prof` dialect:

```mlir
%a1, %w1 = pynqo_prof.enter 40 (%a, %w)          // returns its operands
%r       = torch.aten._int_mm %a1, %w1
%r1      = pynqo_prof.exit 40 (%r)               // returns its operand
```

The markers return their operands, so the data dependence pins them: nothing
that reads `%r1` can run before `exit 40`, and the op cannot start before
`enter 40`. They declare a side effect, so canonicalization and CSE never
delete or merge them.

`torch.aten._assert_tensor_metadata` and ops that produce no value are
skipped. Metadata-only ops (`view`, `unsqueeze`, ...) still get markers; step 3
drops the ones that end up enclosing no work, so the skip list never has to be
maintained by hand.

**2. Lower the markers with the graph.** A conversion pattern carries
`pynqo_prof.enter/exit` from `!torch.vtensor` to `tensor` in torch-to-linalg,
and bufferization treats them as ops that read and forward their operands.
The partitioner (`partition.cpp`) sees the claimed contraction's activation
operand come from a marker instead of the producer, which it already allows:
a claimed op has its constant weight and one other operand.

**3. Turn markers into calls (pynqo-backend CPU pipeline).** One pass at the
memref level, after `convert-bufferization-to-memref` and before
`convert-linalg-to-loops` (`src/export/backend/cpu/toolchain.cpp`):

```mlir
func.call @__prof_enter(%c40, %a_ptr, %w_ptr)    // input buffer pointers
...  the op's linalg ops, or func.call @__delegate__(...)  ...
func.call @__prof_exit(%c40, %r_ptr)             // result buffer pointer
```

- A pair that encloses no linalg op and no `__delegate__` call (a view folded
  into a subview, a dropped assert) is deleted and its id is marked `elided`.
- **The buffer pointers are required.** Intermediate buffers are hoisted
  allocations whose address never escapes, so `llc -O3` may move their loads
  and stores across an opaque call that passes only an id. Passing the input
  pointers to `enter` and the result pointer to `exit` makes them escape, so
  the op's reads happen after `enter` and its writes complete before `exit`.
  (An inline-asm memory clobber in the hook would not help: it only orders
  escaped memory.) The pointers come from
  `memref.extract_aligned_pointer_as_index`.

**Cost table.** The `--profile-aten` rewrite also writes one record per id
into a `profile` table, copied into `compile.json`:

```json
{"id": 40, "op": "aten._int_mm", "entry": "decode",
 "in": [{"shape": [1, 576], "dtype": "si8"}, {"shape": [576, 1536], "dtype": "si8"}],
 "out": [{"shape": [1, 1536], "dtype": "si32"}],
 "ops": 1769472, "bytes_read": 885312, "bytes_written": 6144,
 "device": "npu", "elided": false}
```

- `ops` comes from a per-op formula (contractions `2·M·N·K`, elementwise one
  op per output element, reductions one per input element); ops without a
  formula report `null`, not a guess;
- `bytes_*` is the compulsory traffic (each operand read once, the result
  written once); real traffic is higher when caches thrash, and L3 measures
  the real value for NPU work;
- `device` is filled in after partitioning: `npu` when the pair encloses a
  `__delegate__` call, else `cpu`.

**Export switch.** `src/export/pipeline.py` passes `--profile-aten` to
pynqo-middle-end and the memref-level pass to pynqo-backend when the config
sets `profile: true` or the CLI gets `--profile`; the manifest records
`"profile": "aten"`. Packages built without it contain no markers and no hooks.

**What this measures.** Markers are fusion barriers: `linalg-fuse-elementwise-ops`
no longer fuses across ATen op boundaries, so a profiled package runs the
unfused graph, as PyTorch eager would. The report therefore shows the
profiled step time beside the unprofiled one; the difference is fusion plus
hook overhead. Per-op attribution of a fused production build is a later
level (see "Later").

### L1 runtime: `__prof_enter` / `__prof_exit`

Defined in `src/runtime/delegate/dispatch.c`, which is already loaded with
`RTLD_GLOBAL` before `model.so` and resolves `__delegate__` the same way, so
no new library or loader change is needed.

- Clock: `clock_gettime(CLOCK_MONOTONIC)`, the clock Python's
  `time.perf_counter_ns` reads on Linux, so hook, phase and engine-stage
  events share one timeline. The Cortex-A9 PMU cycle counter (`PMCCNTR`) is
  faster but needs user-mode access enabled by a kernel module; it is a later
  option, not a requirement.
- Storage: per-id aggregates (count, total, min, max) always; an optional
  fixed-size ring of `(id, t_enter, t_exit)` events for a timeline.
- Read out through `prof_reset(n)`, `prof_aggregates(out)` and
  `prof_events(out, max)`, wrapped by `Dispatcher` like the existing
  `delegate_calls`.

### L2: NPU phases (libnpu_rt)

`npu_rt_execute` times its phases with the same clock (`CLOCK_MONOTONIC`) and
keeps, per program index:

| Phase | Time | Bytes |
| --- | --- | --- |
| `pack_a` | `packA` | `m × k` read, A tiles written |
| `stage_b` | each segment's `memcpy` into `stage` | `bBytes` per segment |
| `pl` | `pynqRun`, start to `DONE` | from L3; else the program's static A + B + C bytes |
| `pl_cycles` | `ISA_CYCLES` per segment | |
| `polls` | `ISA_STATUS` reads | |
| `runs` | segment runs (one `START` each) | |
| `instructions` | `ISA_INSTRUCTIONS` per segment: words decoded, `REPEAT` and `END` included, a `REPEAT` once | |
| `unpack_c` | `unpackC` | C planes read, `Kt × m × n × 4` written |

New C API: `npu_rt_profile(enable, event_capacity)`, `npu_rt_phase_stats(index, out)`
and `npu_rt_events(out, max, dropped)` (a bounded timeline of phase events).
Disabled, the only cost is one predictable branch per call. The simulator
backend records the same phases, so host runs exercise the same code path,
but their times are not meaningful for the board.

`pl` wall time against `pl_cycles / f_clk` measures the start and poll
overhead; divided by `runs` (as are `jobs`, `instructions` and `polls`) it is
the per-run cost the report prints as `overhead_ns_per_run`.

### L3: PL performance counters (RTL, ABI 2.1)

New read-only registers after `ISA_VERSION` (0x6C onward), cleared when an ISA
program starts, advertised by a new `Capability.PERF_COUNTERS` bit:

| Register | Counts |
| --- | --- |
| `PERF_RD_BYTES` | bytes accepted on R (`r_valid && r_ready`, × 8) |
| `PERF_WR_BYTES` | bytes accepted on W, weighted by `WSTRB` |
| `PERF_RD_BUSY` | cycles with at least one outstanding read burst |
| `PERF_RD_STALL` | cycles the LSU waits on R data while a job needs operands |
| `PERF_IFETCH` | cycles the front end waits on instruction fetch |
| `PERF_ARRAY_ACTIVE` | cycles the systolic array computes |

These give, per NPU kernel: real DDR bytes and bandwidth, array utilization
(`ARRAY_ACTIVE / ISA_CYCLES`), and whether the PL is starved by memory or by
instruction fetch. The RTL change follows the repository rules: testbench
updates for the counters, `make -C src/test lint sim`, and an ABI minor bump
in `spec/abi.py`. Counters are 32-bit; a single program saturating one is
reported as an overflow, not a wrapped value.

## Derived metrics and report

Per kernel:

```text
bandwidth  = bytes / time            (L3 bytes for NPU kernels when available, else static)
throughput = ops / time
intensity  = ops / bytes
attainable = min(peak_ops, intensity × peak_bandwidth)     from CEILINGS
efficiency = throughput / attainable
bound      = memory if intensity × peak_bandwidth < peak_ops else compute
```

Coverage check: the sum of L1 kernel times plus gaps must match the L0 step
time; the report states the unattributed share (glue code between kernels).

Outputs, written by `src/runtime/share/profile.py` (task-agnostic):

1. `profile.json`: run metadata (package digest, overlay, `CEILINGS`), L0
   stages, per-kernel aggregates with L2/L3 detail, totals.
2. A terminal table: top-N kernels by time with %, GB/s, GOP/s, efficiency
   and bound, plus a roll-up per ATen op type (`aten.mul`, `aten._int_mm`, ...)
   and per device; a per-layer roll-up waits for module paths (see risks).
3. Optional `trace.json` (from the L0 stages, the L1 event ring and the L2
   phases) in the Chrome trace-event format that PyTorch's Kineto writes:
   `ph: "X"` spans in microseconds, with Kineto's categories
   (`user_annotation` for engine stages, `cpu_op` for host work, `kernel` for
   PL runs). Perfetto opens it for a per-token timeline. Holistic Trace
   Analysis (HTA) also reads Kineto traces, but expects GPU-specific fields
   (device properties, streams); whether our traces load in it is unverified
   and is checked in phase 2, not assumed.
4. `profile diff a.json b.json`: per-kernel deltas, for A/B work such as
   vocabulary trimming or #157.

Entry point: a `--profile [DIR]` flag on each task CLI
(`python -m src.runtime.cli llm package --prompt ... --profile out/`),
following the filetree rule that each task's `cli.py` owns its arguments.
`--profile-trace` enables the event ring.

## Overhead budget

An existing SmolLM2-135M export has 14,774 `torch.aten.*` ops across its two
entry points, about 7,400 per step, before elision. At about 1 µs per hook
pair on the Cortex-A9, that is at most about 7 ms against a step of more than
1 s. Hook overhead target: under 2%, measured in phase 2 by timing a package
whose markers are lowered to no-op hooks against the unprofiled package. The
fusion effect (above) is reported separately and has no budget.

## Phases

| Phase | Scope | Changes | Acceptance |
| --- | --- | --- | --- |
| 0 | Board ceilings | Microbenchmarks run on the board (cached/uncached memcpy, NEON GEMM, a synthetic NPU program that streams B); `CEILINGS` in `spec/board.py` | Repeatable within ±3% over 5 runs; values recorded with the overlay and board image used |
| 1 | L0 + L2 | Engine `Profiler`; libnpu_rt phases; `profile.json` and table (no re-export needed) | Fake-clock unit tests for aggregation; disabled path leaves existing tests unchanged; one SmolLM2 board run with the NPU phase split |
| 2a | L1 spike | Insert, lower and call markers on one small graph (linear, add, softmax, one `_int_mm` claimed by the NPU) | The graph exports with `--profile`; claim count unchanged; the backend contract accepts the markers; host run gives one record per non-elided ATen op |
| 2 | L1 | `--profile-aten` in pynqo-middle-end, memref-level pass in pynqo-backend, `compile.json` `profile` table, hooks in `dispatch.c`, roofline report, Chrome trace | FileCheck-style tests for insertion and elision; profiled and unprofiled SmolLM2 packages give bit-identical logits for 24 greedy steps; coverage ≥ 95% of step time; hook overhead < 2%; static op/byte counts match a hand count for one `_int_mm` and one attention block |
| 3 | L3 | RTL counters, testbench, ABI 2.1, driver reads them | `make -C src/test lint sim` passes; counter values match the testbench's expected bytes and cycles; board run reports NPU DDR GB/s |
| 4 | Diff and regressions | `profile diff`; optional CI check on host-sim profiles (op/byte counts, not times) | Diff of two board runs reproduces a known change |

Phase 1 (#172) alone answers how NPU time splits between host copies and PL work,
which is likely where most of today's gap lies. Each phase is one issue and
one PR to `dev`, per the Git rules.

## Risks and open questions

- **Markers through torch-mlir.** The backend contract
  (`torch-backend-to-linalg-on-tensors`) may reject an unknown dialect, and the
  pattern may need a hook in our torch-mlir patch (`patch_torch_mlir.cmake`).
  Phase 2a exists to settle this before phase 2.
- **Profiled build differs from production.** Markers block elementwise
  fusion, so per-op times describe the unfused graph. Bit-identical logits
  are expected (elementwise fusion does not reorder reductions) and are an
  acceptance check, not an assumption.
- **No layer names yet.** `torch.mlir` carries no locations, so records name
  the ATen op and its shapes but not the `nn.Module` path. Adding it means
  writing the FX nodes' `nn_module_stack` into a side table keyed by node order
  in the front end; left for later.
- **Prefill chunk padding.** Prefill runs `chunk` tokens even when fewer are
  real, so its per-op times include padded work; the report labels prefill
  records with the chunk's real length.
- **Clock resolution.** Small CPU kernels (a few µs) are close to the cost of
  `clock_gettime`; report their aggregates only, and validate with PMU cycles
  if needed.
- **Static bytes underestimate CPU traffic.** CPU bandwidth is a lower bound
  until a PMU-based cache-miss count is available (out of scope here).
- **Uncached staging copy.** If phase 1 confirms `stage_b` dominates, the fix
  (resident weights, #157) belongs to that issue, not here.
- **PMU access.** Enabling `PMCCNTR` in user mode needs a kernel module on the
  board image; keep it optional.
- **Host simulator times** are not board times; reports from `sim` runs are
  labeled and must not be compared with board numbers.

## Later

Not part of this plan; recorded so the L1 design does not close them off.

- **Fused-kernel level.** Profile the production (fused) build by wrapping
  each post-fusion linalg op, or by outlining each into its own function
  (IREE and TVM get boundaries this way), and map back to ATen ids through
  locations.
- **AXI Performance Monitor.** A Xilinx APM on `S_AXI_HP0` would give real NPU
  DDR bytes and latency without changing `npu_accelerator`, ahead of the L3
  RTL counters.
- **Counter replay.** Rerun one NPU program several times, reading a different
  counter set each time, as Nsight Compute does when counters are scarce.
