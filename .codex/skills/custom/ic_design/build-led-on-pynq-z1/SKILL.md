---
name: build-led-on-pynq-z1
description: Build, verify, and deploy a PYNQ-Z1 Vivado AXI GPIO overlay that controls the on-board LD0-LD3 LEDs from Python/Jupyter. Use when setting up Vivado and a Basic Tier license for PYNQ-Z1, creating a Zynq Processing System plus AXI GPIO block design, applying XDC pin constraints, generating .bit/.hwh files, troubleshooting bitstream DRC errors, or transferring a custom overlay to a booted PYNQ-Z1 board.
---

# Build LED on PYNQ-Z1

## Overview

Produce the smallest useful PYNQ-Z1 overlay: ARM PS accesses an AXI GPIO register, whose four bits drive LD0-LD3. Treat this as the verified foundation before adding a DMA or NPU accelerator.

Read [references/setup.md](references/setup.md) for installation and board prerequisites. Read [references/vivado-flow.md](references/vivado-flow.md) while building in Vivado. Read [references/deploy-and-debug.md](references/deploy-and-debug.md) before copying files or when a build fails.

## Workflow

1. Confirm the board boots a PYNQ image and is reachable through Jupyter. Install Vivado and activate the no-charge Basic Tier license; do not use a time-limited Enterprise evaluation for this exercise.
2. Create a Vivado RTL project for the PYNQ-Z1 Zynq-7000 device. Add `ZYNQ7 Processing System`, enable `M_AXI_GP0` and `FCLK_CLK0` (100 MHz is a suitable first value), then add `AXI GPIO`.
3. Configure AXI GPIO as one four-bit output channel. Use Connection Automation to add the AXI interconnect/clock/reset links. Make the GPIO interface external.
4. Validate and save the Block Design. The generated top-level signal may be named `gpio_rtl_0_tri_io[3:0]`; inspect the wrapper or DRC error and constrain the *actual* top-level name.
5. Create an XDC mapping the four output bits to LD0-LD3. Do not suppress `NSTD-1` or `UCIO-1`; correct the constraints instead.
6. Create or update the HDL wrapper, then run Generate Bitstream. Copy the `.bit` and `.hwh` to one directory using exactly the same basename.
7. Upload both files to the PYNQ board and load the overlay from a Jupyter Python cell. Write a four-bit value through `axi_gpio_0` to control LEDs.

## Required checks

- Do not use **Generate Block Design** after ordinary edits; open the existing `.bd`, modify it, then **Validate Design**.
- A successful bitstream alone does not prove Python integration. Confirm `Overlay(...).ip_dict` contains `axi_gpio_0`, then run the LED write example.
- If `write_bitstream` fails, inspect the ending of `project.runs/impl_1/runme.log`; use the final DRC errors rather than older out-of-context IP messages.
