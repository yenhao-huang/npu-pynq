## 1. RTL

- [x] 1.1 Add `IN_BYTES` and `s_axis_tkeep` to `npu_matrix_controller` and `npu_matrix_accelerator`, with a row aligner feeding the per-row A and per-column B banks one row-aligned word per cycle.
- [x] 1.2 Validate TLAST and TKEEP per beat and report mismatches as `STREAM_LENGTH`.
- [x] 1.3 Keep the banks' compute read path and the B/compute overlap invariant unchanged.

## 2. Verification

- [x] 2.1 Update the controller, 8x8 and accelerator testbenches to drive packed 64-bit beats.
- [x] 2.2 Cover early and missing TLAST, short and long final TKEEP, partial TKEEP mid-frame, missing B TLAST, and multi-beat frames that end in partial beats.
- [x] 2.3 Assert the beat count `ceil(M*K/8) + ceil(K*N/8)` and the stall-free cycle count `M*ceil(K/8) + K*ceil(N/8) + M*N + M + N + 4` in the 8x8 and scaling testbenches, including 16x16 with two words per B row.

## 3. Overlay

- [x] 3.1 Configure AXI DMA MM2S with 64-bit memory-map and stream widths and set `IN_BYTES = 8` on the accelerator.
- [x] 3.2 Require the new DMA widths, `IN_BYTES` and `s_axis_tkeep` in `verify_overlay.py`, and reject a byte-wide overlay.

## 4. Evidence

- [ ] 4.1 Vivado synthesis, implementation and routed timing at 100 MHz on the self-hosted runner.
- [ ] 4.2 PYNQ-Z1 board run confirming bit-exact results and measured load cycles.
