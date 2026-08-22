## Context

Phase 0 provides executable signed INT8, per-MAC saturating INT32, row-major
matrix, performance, and ABI contracts in `src/test/model/`. No RTL currently
exists. The available host has Icarus Verilog but lacks GNU Make and Verilator,
so focused direct simulation can establish arithmetic evidence while the
repository Make/lint gates remain separately tracked blockers.

Issue `#3` is blocked by Phase 0 Issue `#2`; this branch is deliberately stacked
on `npu/issue2-a` so RTL tests import the exact golden model under review.

## Goals / Non-Goals

**Goals:**

- Produce synthesizable SystemVerilog-2012 accepted by Icarus and intended for
  later Verilator/Vivado validation.
- Make PE arithmetic identical to `mac_int8_int32` at every accepted product.
- Support rectangular arrays and smaller logical shapes through valid masking.
- Make a whole-array stall lossless and deterministic.
- Generate at least 100 seeded matrix cases from the Python golden model and
  consume tracked vectors in self-checking RTL simulation.

**Non-Goals:**

- Add local SRAM, tiling controller, AXI4-Stream protocol, AXI4-Lite registers,
  DMA, MMIO, or autonomous job scheduling.
- Claim timing closure, utilization, Vivado elaboration, bitstream generation,
  or PYNQ board behavior.
- Change Phase 0 arithmetic, ABI, layouts, or performance equations.

## Decisions

### Use an output-stationary PE with independent forwarded streams

Each PE owns one INT32 accumulator. On an enabled edge it registers A to the
right and B downward, propagates their valid bits independently, and performs a
MAC only when both incoming valids are high. A clear edge zeros accumulator and
pipeline state. The array output can therefore be read directly without a
separate drain network.

Alternative considered: weight-stationary PEs. Rejected for Phase 1A because
it requires weight-load state and complicates a minimal proof of symmetric
matrix multiplication; it can be reconsidered through a spec change if ResNet
performance evidence justifies it.

### Use a single global enable as the first backpressure boundary

`enable=0` freezes every PE register and accumulator. Upstream must hold edge
inputs and valid bits, and its logical skew counter advances only on enabled
edges. This provides lossless backpressure without combinational ready chains
across the array.

Alternative considered: ready/valid per PE. Rejected because combinational
backpressure through a two-dimensional mesh is unnecessary for the first
vertical slice and creates timing and deadlock risks before a controller exists.

### Flatten array ports at the module boundary

The module exposes packed one-dimensional A, B, valid, and accumulator buses.
Generate loops slice them into internal two-dimensional nets. This preserves a
clear row-major ABI and avoids inconsistent simulator support for unpacked
array ports while retaining parameterized rows and columns.

### Define skew timing in active steps

Logical step advances only when enable is high. A[m,k] enters row m at step
m+k and B[k,n] enters column n at step n+k. Register forwarding causes both to
meet at PE [m,n] on step m+n+k. The last required MAC occurs at
M+N+K-3. Reset and clear edges are not active steps.

This schedule is intentionally upstream-visible; Phase 1B may implement it in
a controller without changing PE/array arithmetic.

### Implement saturation with one extra signed sum bit

The PE sign-extends the current INT32 accumulator and exact INT16 product to 33
bits, compares the sum against sign-extended INT32 bounds, and selects the
upper bound, lower bound, or low 32 sum bits. This mirrors Python saturation
without relying on simulator overflow behavior.

Alternative considered: detect overflow from sign bits after a 32-bit add.
Rejected because the product must first be widened and the 33-bit form is more
direct to review against Phase 0.

### Use tracked text vectors with a deterministic generator

`src/test/vectors/generate_systolic_vectors.py` writes a compact line-oriented
file containing case dimensions, flattened signed A/B values, expected INT32
outputs, and a deterministic stall step. A Python unit test regenerates into a
temporary path and compares bytes to the tracked fixture. The SystemVerilog
randomized testbench reads that fixture and reports the case/coordinate on any
mismatch.

The first randomized harness fixes the physical array at 2x2 and varies logical
M/N from 1 to 2 and K from 1 to 8. Parameterization is separately elaborated
and functionally tested with a 2x3 deterministic instance.

Alternative considered: DPI or cocotb. Rejected for now because neither is
configured in the repository, whereas a tracked vector boundary works with the
available Icarus installation and remains consumable by future simulators.

### Keep testbenches individually discoverable

RTL lives in `src/hw/rtl/systolic_array/`; self-checking testbenches live in
the matching `src/hw/tb/systolic_array/` and use `tb_<module>.sv` names. The
Makefile continues to discover them and may receive a per-test top/source fix
if wildcard compilation causes duplicate testbench elaboration.

## Risks / Trade-offs

- [Per-MAC saturation adds a 33-bit compare path] -> Keep it contract-correct
  in Phase 1A and measure timing/resource cost in Phase 1B before optimization.
- [Global stall reduces fine-grained throughput] -> It is deterministic and
  deadlock-free; add elastic boundaries only with controller-level evidence.
- [Tracked vectors can drift from generator] -> Byte-for-byte regeneration is
  a required Python test and the seed/case count live in the file header.
- [Stacked changes make the PR diff temporarily large] -> Keep Issue #3 commits
  isolated and sync `dev` after PR #11 and PR #12 merge.
- [Icarus passing is weaker than Verilator/Vivado] -> Record direct simulation
  separately and leave unavailable lint/synthesis/timing gates blocked.

## Migration Plan

1. Merge repository foundation PR #11 and Phase 0 PR #12 into `dev` after
   review, then sync this branch normally without history rewriting.
2. Land Phase 1A RTL, vectors, tests, and OpenSpec artifacts through its own PR.
3. Make Phase 1B controller/DMA consume the array edge/enable/clear interface.
4. If Phase 1A is rejected, revert its isolated commits; no board artifact or
   software ABI migration is required.
