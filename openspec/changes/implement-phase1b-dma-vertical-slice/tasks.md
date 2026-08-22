## 1. AXI4-Lite Control Plane

- [x] 1.1 Add a self-checking public-port AXI4-Lite testbench for ABI identity, independent AW/W arrival, WSTRB, stable B/R responses, reserved accesses, configuration writes, START, BUSY_START, SOFT_RESET, and sticky status/errors; record that it fails before the control RTL exists.
- [x] 1.2 Implement the ABI v1 AXI4-Lite register block and verify the focused control testbench passes under Icarus with no error/failure markers.

## 2. Matrix Stream and Compute Controller

- [x] 2.1 Add failing self-checking tests for valid A/B frames, physical-limit validation, signed endpoint math, M/N masking, input/output stalls, early and missing TLAST, timeout, cycle stability, and output TLAST.
- [x] 2.2 Implement bounded A/B buffering, systolic skew scheduling, logical-lane serialization, frame validation, timeout/error recovery, and the public accelerator top; verify all focused controller tests pass.
- [x] 2.3 Add an integrated AXI-Lite plus AXI4-Stream testbench that submits the exact board-smoke matrix through public ports and verifies C=[[636,-891],[-19,29]], DONE without ERROR, nonzero stable cycles, and output backpressure.
- [x] 2.4 Register scheduled controller operands and accepted PE products, add the required drain steps, and update the latency/reset/stall contracts and deterministic tests; verify all seven Icarus testbenches pass and the routed 100-MHz implementation meets timing.

## 3. PYNQ Runtime

- [x] 3.1 Add failing host tests for HWH/IP discovery, ABI negotiation, physical limits, signed INT8 shape/dtype validation, 64-byte buffer rules, non-aliasing, and invalid-job no-write/no-DMA behavior.
- [x] 3.2 Implement the dependency-injected runtime and real PYNQ overlay factory; verify metadata, preflight, signed conversion, and Phase 0 ABI parity tests pass without importing `pynq` on the host.
- [x] 3.3 Add fake-MMIO/DMA tests for the exact S2MM→START→A→B→status sequence, cache flush/invalidate, finite deadlines, ABI errors, DMA failures, soft-reset recovery, and no result on failure; verify the focused runtime suite passes.

## 4. Reproducible Vivado Overlay

- [x] 4.1 Implement repository-relative `npu_matrix` Vivado Tcl for PS7, GP0 control, HP0 DDR access, simple AXI DMA, module-reference accelerator, resets, interrupts, fixed addresses, reports, and same-build artifacts; verify a narrow Vivado batch parse/elaboration command succeeds.
- [x] 4.2 Run the clean full Vivado batch build and verify synthesis/implementation success, zero DRC errors, fully routed design, WNS >= 0, zero setup-failing endpoints, expected utilization, successful bitstream generation, and exact accelerator/DMA HWH metadata.
- [x] 4.3 Add source-controlled provenance/HWH verification that rejects stale or mismatched artifacts; verify it passes for the current generated `npu_matrix.bit`/`npu_matrix.hwh` pair and generated files remain ignored.

## 5. Physical Board Vertical Slice

- [x] 5.1 Implement a board smoke entry point that records non-secret provenance/metadata, runs the signed 2x2 endpoint matrix through real PYNQ buffers/MMIO/DMA/PL, checks every result/status/cycle assertion, and prints only the specified PASS marker on success; verify host syntax and failure-path tests pass.
- [ ] 5.2 Use the repository board-transfer workflow with the current matching artifacts, run the smoke command on PYNQ-Z1, and record the exact `PASS: NPU DMA matrix vertical slice` output plus PYNQ/IP/address evidence.

## 6. Integration and Handoff

- [x] 6.1 Run all Python tests, every direct Icarus simulation, strict OpenSpec validation, whitespace/path/generated-artifact/secret/human-document scans, and record exact pass/fail/blocked evidence in `STATE.md`.
- [ ] 6.2 Run repository `make -C src/test lint sim` and CI for the published head; keep this task incomplete if GNU Make, Verilator, or CI evidence is unavailable or failing.
- [x] 6.3 Inspect numeric/ABI/runtime/Tcl/HWH parity and the complete diff, create issue-linked Conventional Commits, publish one draft PR to `dev`, and preserve all unresolved Vivado/board/dependency gates without merging or closing Issue #4.
