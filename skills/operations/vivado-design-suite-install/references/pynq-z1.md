# PYNQ-Z1 overlay prerequisites

PYNQ-Z1 boots the PYNQ software environment from a microSD card. The board uses the Zynq processing system for its operating system and a programmable-logic overlay for FPGA hardware.

When Vivado produces an overlay, retain both files with the same base name:

```text
my_overlay.bit
my_overlay.hwh
```

PYNQ reads the `.hwh` hardware-handoff file to discover IP, clocks, interrupts, and memory-mapped interfaces when loading the `.bit` file. Keep the PYNQ image, board files, and Vivado version compatible with the overlay project being built; when rebuilding an existing PYNQ design, use the tool version documented by that release.

Sources: [PYNQ-Z1 setup guide](https://pynq.readthedocs.io/en/v2.7.0/getting_started/pynq_z1_setup.html), [PYNQ overlay design](https://pynq.readthedocs.io/en/v3.1/overlay_design_methodology/overlay_design.html).
