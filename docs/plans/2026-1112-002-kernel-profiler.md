# Kernel profiler plan

| Field | Value |
| --- | --- |
| Plan ID | 2026-1112-002 |
| Topic | Per-kernel time and bandwidth profiling for `pynqo-package/2` |
| Date | 2026-11-12 |
| Status | Draft |
| Based on | `origin/dev` at `95c35f7` |
| Related | #115 (per-stage timing, written for the deprecated JIT runtime), #157 (asynchronous delegates) |
| Issues | #172 (phase 1), #195 (phase 2: L1 on fused kernels) |

## Goal

For one inference run (an LLM prefill/decode step, or a classifier forward) on
the PYNQ-Z1, report, **for every kernel**:

- wall time (total, count, min, max, share of the step);
- bytes moved and the resulting effective bandwidth (GB/s);
- operations and throughput (GOP/s);
- where it sits against the board's roofline (compute-bound or memory-bound,
  percentage of the attainable ceiling).

A *kernel* is **one operation of the CPU graph as `model.so` runs it**: a
linalg operation after elementwise fusion (one loop nest), a tensor operation
that copies (`concat`, `insert_slice`, `pad`), a copy bufferization adds, or a
`__delegate__` call. This is the production build's own granularity, as
IREE's dispatches and TVM's fused functions are; a fused kernel can cover
several ATen ops. Each NPU call is further split into its host-side phases
and its PL time (L2).

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
L1 kernel        generic(exp,divf)#57 | concat#58 | delegate 3#59 ...       (model.so probes)
L2 npu phases    packA | stage weights (per segment) | pl run | poll | unpackC  (libnpu_rt)
L3 pl counters   AXI read/write bytes, busy and stall cycles, array-active cycles  (RTL)
```

### L0: engine stages (Python only)

`ExportedLLM.generate` and the classifier engine take an optional `Profiler`
and record `tokenize`, `prefill`, each `decode`, `sample` and `detokenize`
with `time.perf_counter_ns`. The same profiler collects L1–L3 records for the
step it encloses, so every kernel record carries its step (prefill chunk or
decode position).

### L1: probes around every fused kernel (#195)

As ExecuTorch (debug handles at export, events at run time, joined by a table)
and IREE (dispatches formed in `flow`, instrumentation materialised in
codegen), naming a kernel and timing it are separate steps in separate stages
(`docs/rules/lib/export_stages.md`).

**1. Name the kernels (pynqo-middle-end, linalg on tensors).**
`pynqo-assign-kernel-ids` runs last, after `linalg-fuse-elementwise-ops`, and
sets `pynqo.kernel_id = <i> : i32` on every kernel of the CPU graph in program
order, delegate calls included. Views (`extract_slice`, `expand_shape`,
`collapse_shape`) and scalar reads move no data and are not kernels. It is
whole-graph and target-agnostic, so every CPU target shares the numbering.
The attribute is always written: lowering to loops drops it, so it changes no
code (checked: `model.so` is byte-identical with and without it).

**2. Probe them (pynqo-backend `--profile-kernels`, per CPU target).**
`src/export/backend/cpu/kernel_probes.cpp`, around bufferization in
`cpu/toolchain.cpp`:

```text
probeKernels    tensors, before bufferization: each numbered operation is wrapped in
                func.call @__prof_enter(%id) / @__prof_exit(%id). The calls stay in
                place through bufferization, so the allocations and copies it adds for
                an operation (a concat's copies, a copy before an in-place write) are
                timed with it, whether or not the attribute survives.
bufferization   unchanged
probeCopies     memrefs, after bufferization: each memref.copy still outside a window
                (a result copied into its out-parameter, a delegate's input staged in
                its buffer) gets a window and a new id after the last.
to LLVM         unchanged
```

The delegate call carries its number through `cpu/delegate_calls.cpp`.
`kernels.json` beside each `model.so` describes every id (kind `kernel`,
`delegate` or `copy`; op; entry point; operand and result types; a linalg
op's body operations; the delegate id), and `compile.json` lists it under
`kernels`. Without the flag the pipeline is the single string it was and the
package is unchanged.

**Export switch.** `python -m src.export.cli build ... --profile-kernels`
(`pipeline.export_package(profile_kernels=True)`). A probed package is for
profiling, not deployment.

**Ordering caveat.** The probes pass only the id. `llc -O3` may move loads and
stores of buffers whose address never escapes across an opaque call, so a
little work can land in the neighbouring window. Passing the buffer pointers
(`memref.extract_aligned_pointer_as_index`) to the probes would pin them; it
is not done yet and is checked against the disassembly and the coverage
figure first (Risks).

### L1 runtime: `__prof_enter` / `__prof_exit`

`src/runtime/share/kernel_probes.c`, wrapped by `src/runtime/share/kernels.py`
(`KernelProbes`). `ExportedPackage` loads it with `RTLD_GLOBAL` before a
probed `model.so`, as it loads the delegate dispatcher.

- Off until profiling starts: each probe is then one load and one branch.
- Clock: `clock_gettime(CLOCK_MONOTONIC)`, the clock Python's
  `time.perf_counter_ns` reads on Linux, so probe, phase and engine-stage
  events share one timeline. The Cortex-A9 PMU cycle counter (`PMCCNTR`) is a
  later option.
- Storage: per-id calls, total, min, max; with an event capacity, `(id, begin,
  end)` events until full, counting the rest as dropped.
- `share/profile.py` collects them on detach: `profile.json` gains `kernels`
  (per kernel, slowest first; time per kind; `coverage` = probed time over
  package-call time), the table a top-N kernel list, and `trace.json` a
  `cpu kernels` track.

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
| `unpack_c` | `unpackC` | C planes read, `Kt × m × n × 4` written |

New C API: `npu_rt_profile(enable, event_capacity)`, `npu_rt_phase_stats(index, out)`
and `npu_rt_events(out, max, dropped)` (a bounded timeline of phase events).
Disabled, the only cost is one predictable branch per call. The simulator
backend records the same phases, so host runs exercise the same code path,
but their times are not meaningful for the board.

`pl` wall time against `pl_cycles / f_clk` measures the start and poll
overhead.

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
| 2 | L1 (#195) | `pynqo-assign-kernel-ids` in pynqo-middle-end; `--profile-kernels` probes in pynqo-backend; `kernels.json`; `kernel_probes.c`; kernel table and trace in `profile.py` | Host tests: ids in program order, delegate calls numbered, device modules not; `model.so` byte-identical without the flag; every id probed and described; probed and plain packages give identical outputs, every kernel timed once per call. Board: SmolLM2 logits identical for 24 greedy steps; coverage ≥ 95% of package-call time; probe overhead < 2% |
| 2b | Roofline | Static op and byte counts per kernel from `kernels.json`; roofline columns | Counts match a hand count for one contraction and one attention block |
| 3 | L3 | RTL counters, testbench, ABI 2.1, driver reads them | `make -C src/test lint sim` passes; counter values match the testbench's expected bytes and cycles; board run reports NPU DDR GB/s |
| 4 | Diff and regressions | `profile diff`; optional CI check on host-sim profiles (op/byte counts, not times) | Diff of two board runs reproduces a known change |

Phase 1 (#172) alone answers how NPU time splits between host copies and PL work,
which is likely where most of today's gap lies. Each phase is one issue and
one PR to `dev`, per the Git rules.

## Risks and open questions

- **Probe ordering.** See the caveat under L1: only the id is passed. If the
  disassembly or a low coverage shows work moving between windows, pass the
  buffer pointers.
- **No layer names yet.** `model.mlir` carries no locations, so a kernel is
  named by its op, body operations and types, not by its ATen op or
  `nn.Module` path. Plan 2026-1112-004's provenance step (FX locations kept to
  `model.mlir`) fills `location` in `kernels.json`.
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

- **ATen-op level.** Time the unfused graph per ATen op, as the PyTorch
  profiler does: marker ops threaded through the torch dialect
  (`pynqo_prof.enter/exit` returning their operands), carried through
  torch-to-linalg and turned into calls on memrefs. They block fusion, so they
  describe the unfused graph; the fused-kernel probes (L1) describe what runs.
- **Kernel outlining.** Once plan 2026-1112-004 outlines each kernel into its
  own function, probes can move to the call sites, as TVM's debug executor
  times fused functions; the kernel ids and `kernels.json` stay.
- **AXI Performance Monitor.** A Xilinx APM on `S_AXI_HP0` would give real NPU
  DDR bytes and latency without changing `npu_accelerator`, ahead of the L3
  RTL counters.
- **Counter replay.** Rerun one NPU program several times, reading a different
  counter set each time, as Nsight Compute does when counters are scarce.
