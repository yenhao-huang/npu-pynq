## Why

Phase 1A proves the signed saturating systolic datapath in simulation, but it
cannot yet be controlled by the Zynq processing system or move matrix payloads
through DDR. Phase 1B must turn that isolated compute core into the first
reproducible PYNQ-Z1 hardware/software vertical slice while preserving the
Phase 0 ABI and objective evidence boundaries.

## What Changes

- Add an ABI v1 AXI4-Lite control plane and an AXI4-Stream matrix datapath
  around the Phase 1A systolic array, including independent AXI handshakes,
  backpressure, TLAST validation, timeout handling, sticky status, and stable
  error reporting.
- Add a self-checking integrated testbench that drives the public AXI-Lite and
  AXI-Stream interfaces and covers success, stalls, invalid jobs, malformed
  streams, busy-start, reset, and timeout behavior.
- Add repository-relative Vivado Tcl that recreates the PYNQ-Z1 processing
  system, AXI DMA, accelerator, clocks/resets, interrupt wiring, and fixed
  address map, then emits auditable utilization, timing, DRC, address, bitstream,
  and HWH evidence outside Git.
- Add a host-testable PYNQ runtime that validates buffers, discovers IP through
  HWH metadata, sequences DMA and MMIO safely, enforces finite timeouts, and
  returns signed row-major INT32 results.
- Add a physical-board smoke entry point whose PASS marker proves one real
  DDR-to-DMA-to-NPU-to-DMA-to-DDR matrix transaction using a matching bit/HWH
  pair.

## Capabilities

### New Capabilities

- `matrix-accelerator-interface`: ABI v1 AXI4-Lite job control and AXI4-Stream matrix input/output behavior around the systolic array.
- `pynq-overlay-build`: Reproducible PYNQ-Z1 Vivado block-design generation, address assignment, artifacts, and implementation evidence.
- `pynq-matrix-runtime`: Host and board runtime behavior for ABI negotiation, buffer validation, DMA sequencing, timeout/error propagation, and signed results.
- `matrix-board-vertical-slice`: Objective physical-board acceptance for a matching overlay and an end-to-end matrix transaction.

### Modified Capabilities

- `signed-processing-element`: Register an accepted product before the saturating accumulator update so the XC7Z020 datapath can close at 100 MHz; reset, clear, and stall cover this product stage.
- `systolic-array`: Require one enabled product-pipeline drain step after the final skew step before accumulator results are final.

The signed INT8/per-MAC saturating INT32 numeric result, operand forwarding,
global-stall behavior, and public array ports remain compatible. Only internal
compute latency increases by one enabled step.

## Impact

- RTL and testbench sources under `src/hw/rtl/` and `src/hw/tb/`.
- Project-generating Tcl under `src/hw/vivado_tcl/` and any required source XDC
  under `src/hw/constraints/`.
- Runtime modules and host tests under `src/runtime/` and `src/test/tests/`.
- Existing simulation discovery through `src/test/Makefile`.
- Local/self-hosted Vivado, generated overlay artifacts, HWH metadata, PYNQ-Z1
  DDR/DMA/MMIO behavior, and the board at the repository-defined target.
- No generated Vivado project, bitstream, HWH, report, cache, credential, or
  board output is committed.
