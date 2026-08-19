---
name: build-mac-npu-on-pynq-z1
description: Build, verify, synthesize, deploy, and debug the repository's signed INT8 MAC accelerator for PYNQ-Z1 from pure SystemVerilog through AXI4-Lite and PYNQ Python MMIO. Use when working in mount/mac_npu on MAC arithmetic, self-checking XSIM tests, register-map behavior, Vivado Zynq block design or bitstream/HWH generation, Overlay/MMIO Python control, Ethernet deployment, hardware smoke tests, or progression toward vector MAC, DMA, GEMM, and compiler/runtime integration.
---

# Build MAC NPU on PYNQ-Z1

Work from `npu_repo_in_pynq` and keep deployable files under `mount/mac_npu/`.
Advance only when the current phase gate has objective evidence. Never treat a
simulation result as proof that board MMIO works.

## Select the reference

- Read [end-to-end-runbook.md](references/end-to-end-runbook.md) for every new
  implementation or continuation. It defines phases, commands, artifacts,
  gates, checkpoints, and completion criteria.
- Read [installation/icarus-verilog-windows.md](references/installation/icarus-verilog-windows.md)
  when installing, repairing, upgrading, locating, or validating Icarus
  Verilog (`iverilog`/`vvp`) on Windows. Keep package-installation procedures
  under `references/installation/` rather than scattering them across build or
  verification references.
- Read [architecture.md](references/architecture.md) before modifying signed
  arithmetic, accumulator semantics, pipeline behavior, or the roadmap.
- Read [rtl-and-axi-contract.md](references/rtl-and-axi-contract.md) before
  editing `mac_unit.sv`, `mac_axi_lite.sv`, either SystemVerilog testbench, or
  the register map.
- Read [verification.md](references/verification.md) before running or changing
  Python tests, XSIM, synthesis checks, or acceptance evidence.
- Read [vivado-overlay-flow.md](references/vivado-overlay-flow.md) before
  changing the block design Tcl, building `.bit/.hwh`, or diagnosing Vivado.
- Read [pynq-integration.md](references/pynq-integration.md) before modifying
  `mac_mmio.py`, overlay loading, IP discovery, or Python call semantics.
- Read [board-deploy-and-debug.md](references/board-deploy-and-debug.md) before
  syncing, configuring Ethernet, using SSH, running the hardware smoke test, or
  deciding that board access is blocked.
- Read [rules/state-rules.md](references/rules/state-rules.md) before changing
  `STATE.md`.

## Non-negotiable sequence

1. Freeze the arithmetic and register contracts.
2. Pass dependency-free Python reference/driver tests.
3. Pass the self-checking AXI4-Lite XSIM test.
4. Build matching `mac_npu.bit` and `mac_npu.hwh`; inspect address and timing.
5. Establish physical network reachability to the PYNQ-Z1.
6. Sync files, load the overlay, exercise real MMIO, and capture the board PASS.

Do not start DMA, matrix tiling, or compiler lowering before step 6 succeeds.

## Completion contract

The scalar MMIO milestone is complete only when the PYNQ board runs
`hardware_smoke_test.py` and prints:

```text
PASS: PYNQ MMIO wrote a/b/clear/start and read accumulator
```

The evidence must cover clear-to-zero, positive accumulation, signed negative
input, signed INT8 endpoints, done polling, and signed INT32 readback.
