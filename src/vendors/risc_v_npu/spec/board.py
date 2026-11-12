"""The PYNQ-Z1's CPU: 32-bit hard-float Linux on a Cortex-A9 with NEON.

Copied from src/vendors/npu_pynq/spec/board.py: risc_v_npu keeps its own copy
rather than importing npu_pynq (plan 2026-1112-003 section 6.2). Keep the two
in step when the board changes.
"""

CPU = dict(model='arm-cortex-a9', triple='armv7a-unknown-linux-gnueabihf', cpu='cortex-a9',
           features='+neon,+vfp3', index_bits=32, float_abi='hard')

# The NPU's clock: FCLK_CLK0 (PCW_FPGA0_PERIPHERAL_FREQMHZ in
# src/hw/vivado_tcl/npu_accelerator/build_overlay.tcl). ISA_CYCLES counts it.
# The CVA6 build runs CVA6 in the same single 100 MHz domain.
PL_CLOCK_HZ = 100_000_000
