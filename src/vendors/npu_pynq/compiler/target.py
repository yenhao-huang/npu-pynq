"""The PYNQ-Z1 CPU target, as the fields of src/export's ``Target``."""

# The PYNQ-Z1's Cortex-A9 runs 32-bit hard-float Linux with NEON.
CPU = dict(name="pynq", triple="armv7a-unknown-linux-gnueabihf", cpu="cortex-a9",
           features="+neon,+vfp3", index_bits=32, library="model.so", float_abi="hard")
