"""The PYNQ-Z1's CPU: 32-bit hard-float Linux on a Cortex-A9 with NEON."""

CPU = dict(model='arm-cortex-a9', triple='armv7a-unknown-linux-gnueabihf', cpu='cortex-a9',
           features='+neon,+vfp3', index_bits=32, float_abi='hard')
