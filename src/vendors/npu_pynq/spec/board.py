"""The PYNQ-Z1's CPU: 32-bit hard-float Linux on a Cortex-A9 with NEON."""

CPU = dict(model='arm-cortex-a9', triple='armv7a-unknown-linux-gnueabihf', cpu='cortex-a9',
           features='+neon,+vfp3', index_bits=32, float_abi='hard')

# The NPU's clock: FCLK_CLK0 (PCW_FPGA0_PERIPHERAL_FREQMHZ in
# src/hw/vivado_tcl/npu_accelerator/build_overlay.tcl). ISA_CYCLES counts it.
PL_CLOCK_HZ = 100_000_000
