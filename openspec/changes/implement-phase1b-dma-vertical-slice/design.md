## Context

See `proposal.md` for motivation. Phase 0 freezes signed INT8/per-MAC
saturating INT32 arithmetic and ABI v1; Phase 1A supplies a verified 2x2-capable
parameterized systolic array. The repository has no current production runtime,
AXI wrapper, DMA design, or overlay Tcl. Source must fit a Zynq-7020, generated
Vivado output cannot enter Git, and simulation, implementation, HWH, and board
evidence are separate gates.

The established tree overrides the generic development convention: design RTL,
testbench, and Tcl use matching `npu_matrix` directories under `src/hw/`; board
software lives in `src/runtime/`; host tests remain under `src/test/tests/`.

## Goals / Non-Goals

**Goals:**

- Realize the complete ABI v1 control lifecycle for one bounded physical matrix
  job and preserve AXI correctness under independent channels/backpressure.
- Move A, B, and C through real PS DDR and AXI DMA rather than per-element MMIO.
- Make every source, host, Vivado, metadata, and board gate reproducible and
  independently auditable.
- Provide a runtime boundary that Phase 1C can call and Phase 2A can extend with
  logical tiling without changing the board transport.

**Non-Goals:**

- Logical matrices larger than the 2x2 physical output tile, K larger than 256,
  non-dense views, batched GEMM, convolution lowering, requantization, ResNet,
  Transformer, and performance tuning.
- Scatter-gather DMA, cache-coherent AXI ports, interrupts as the only completion
  mechanism, committing bitstreams/HWH/reports, or changing host network/tool
  configuration.

## Decisions

### One bounded job controller wraps the existing array

Create `src/hw/rtl/npu_matrix/` with a protocol-correct AXI4-Lite register
block, a matrix stream/controller, and a public accelerator top. The controller
buffers at most 2x256 A bytes and 256x2 B bytes, then drives the existing
`npu_systolic_array` with the same skew schedule already proven by Phase 1A.
It serializes only logical C lanes onto the output stream.

This preserves the arithmetic core boundary and makes stream framing/errors
testable without Vivado IP. Directly placing the array behind DMA was rejected:
the Phase 0 A-then-B frames require reuse, framing checks, masking, and status
that the bare array does not provide.

Vivado placement exposed first the PE multiplier-plus-saturating-adder path and
then the controller LUTRAM-read-plus-multiplier path as 100-MHz timing failures.
Each PE therefore registers an accepted signed product and applies it to the
saturating accumulator on the next enabled step. The controller also registers
scheduled edge operands between its LUTRAM buffers and the array, and remains
in compute long enough to drain both added stages before exposing results.
Reset, synchronous clear, and global stall cover the product stage, so the
numeric result and lossless-stall contract remain unchanged while RAM lookup,
multiplication, and accumulation occupy separate timing stages. The PE marks
the signed multiplier for DSP implementation so array growth consumes the
XC7Z020 DSP48 budget instead of scaling the product in general-purpose LUTs;
the routed utilization report remains authoritative.

### Physical limits are explicit HWH parameters

Phase 1B uses `ROWS=2`, `COLUMNS=2`, and `MAX_K=256`. These are implementation
limits, not a narrowing of the ABI field encoding. The RTL validates them, the
module parameters are emitted in HWH, and runtime rejects larger physical jobs.
Phase 2A owns logical tiling and may issue many bounded jobs through the same
interface.

Adding new ABI registers was rejected because 0x3C-0xFF is frozen reserved in
Phase 0. Silently accepting larger jobs was rejected because it could corrupt
DMA output.

### Two input frames and one output frame

MM2S TDATA is 8 bits: one signed element per beat. Runtime sends A and B as two
simple-mode DMA transfers, so each produces an unambiguous TLAST. S2MM TDATA is
32 bits: one signed accumulator per beat, TLAST on C[M*N-1]. The controller
checks both input frame boundaries and holds output stable under backpressure.

A packed 32-bit input stream was rejected because partial final beats and TKEEP
would add alignment behavior not frozen by Phase 0. A single concatenated input
buffer was rejected because it loses the A/B TLAST boundary.

### AXI-Lite is a standalone tested block

The register block buffers AW and W independently, allows one outstanding write
and one outstanding read response, honors WSTRB, and emits one-cycle command
pulses. Configuration writes are accepted only while idle; read-only and
reserved writes are ignored. The controller owns sticky status, the first error,
and the 64-bit cycle count.

Using a friendly same-cycle AW/W slave was rejected because real PS/interconnect
traffic does not guarantee channel alignment. Generating a packaged custom IP
was rejected for this phase because a module-reference block design is fully
source-controlled and avoids checked-in IP products.

### Runtime uses dependency injection and HWH discovery

`src/runtime/npu.py` keeps `pynq` imports inside the real-overlay factory while
the main runtime accepts injected overlay/IP/channel/allocator/clock objects.
It checks Phase 0 ABI constants, HWH parameters, shapes, dtype, physical ranges,
and finite deadlines before or during the fixed transfer sequence. It performs
best-effort SOFT_RESET and bounded DMA recovery on failure and never returns C
unless all status and length conditions succeed.

Importing `src/test/model` from production was rejected. Runtime constants will
be defined in the runtime boundary and parity-tested against the Phase 0 model
until a later exporter change promotes a shared contract module.

### One Tcl build owns topology and evidence

`src/hw/vivado_tcl/npu_matrix/build_overlay.tcl` uses the exact target part,
module-reference RTL, PS7, simple AXI DMA, control and DDR interconnects,
processor reset, and interrupt concat. It forces accelerator 0x43C00000/64 KiB
and DMA 0x40400000/64 KiB, launches implementation through write_bitstream,
writes reports/address evidence, and copies same-build `npu_matrix.bit` and
`npu_matrix.hwh` under ignored output paths.

The PS FCLK target is 100 MHz; the actual HWH frequency and routed timing report
are authoritative. If timing fails, the build remains failed rather than
silently lowering the clock. IP versions are resolved from the installed Vivado
catalog and recorded in evidence instead of hard-coded to an unverified version.

### Verification layers remain non-substitutable

The integrated SystemVerilog testbench drives only public AXI ports and covers
control, two frames, signed math, masking, backpressure, malformed TLAST,
busy-start, invalid configuration, timeout, and soft reset. Host tests cover
runtime sequencing and failure recovery. Vivado reports prove implementation;
HWH inspection proves metadata; only the PYNQ smoke PASS proves the full board
path.

## Risks / Trade-offs

- [Hand-written AXI4-Lite logic can deadlock under unusual channel order] → Use
  independent AW/W buffers, response stability assertions, and deliberately
  skewed testbench transactions.
- [Register-array buffers may infer LUT RAM instead of BRAM] → Write synchronous
  storage patterns, inspect utilization, and fail the resource review if the
  result threatens Zynq-7020 budget.
- [A DMA or hardware timeout can leave a channel non-idle] → Apply bounded
  recovery, surface the failure, and require overlay reload before reuse when
  recovery cannot be proven.
- [Vivado module-reference parameter names may not appear in PYNQ `ip_dict` as
  expected] → Inspect generated HWH and make runtime construction fail closed;
  do not fall back to guessed limits.
- [100-MHz timing may fail after integration] → Treat routed WNS as a required
  gate and revise pipelining/Tcl in this same issue rather than claiming a
  lower-layer PASS.
- [The 2x2 physical job limit is not a complete user-facing GEMM] → Keep the
  limitation explicit; Phase 1C demonstrates it and Phase 2A adds logical tiling.

## Migration Plan

1. Land repository foundation, Phase 0, and Phase 1A dependencies into `dev`.
2. Merge the source-only Phase 1B PR after open-source checks and review; keep
   Vivado and board gates explicit if infrastructure is unavailable.
3. On a licensed host, regenerate the overlay from Tcl and archive reports plus
   provenance outside Git.
4. Verify HWH and same-build bitstream, then deploy through the repository board
   transfer workflow and run the exact smoke command.
5. Roll back source by reverting the Phase 1B commits; roll back a deployed
   board by loading the previously proven matching overlay pair.

## Open Questions

- The installed Vivado catalog may select different compatible PS7, DMA,
  interconnect, or reset IP revisions; the Tcl evidence will record them and
  the generated HWH remains authoritative.
