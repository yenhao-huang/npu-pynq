---
name: vivado-design-suite-install
description: Install and configure AMD Vivado Design Suite on Windows for PYNQ-Z1 / Zynq-7000 development, including the correct minimal device selection and node-locked license import. Use when a user asks to download, install, license, configure, or troubleshoot Vivado for a PYNQ-Z1.
---

# Vivado Design Suite Installation for PYNQ-Z1

Use this skill to guide a Windows user through installing Vivado for a PYNQ-Z1.

## Workflow

1. Read `references/amd-installer.md` and confirm the user is using the official AMD Windows Unified Web Installer.
2. In the installer, select **Vivado**. In customization, retain **Vivado** and **Vitis HLS**; select only **SoCs → Zynq-7000** under Devices. Do not select unrelated device families.
3. Treat MATLAB/Simulink, embedded software, power-analysis, and offline-documentation components as optional. Use `references/selection-guide.md` for the exact choices.
4. If license setup is required, use `references/licensing.md`. The license host is the Windows PC running Vivado, never the PYNQ-Z1. Do not request, handle, or enter passwords for the user.
5. After installation, verify that Vivado starts, that `xc7z020clg400-1` is selectable, and that the license manager reports the intended license as valid when applicable.
6. For a PYNQ overlay, load the appropriate PYNQ board files and follow `references/pynq-z1.md`; keep the generated `.bit` and matching `.hwh` together.

## Guardrails

- Use current AMD documentation and the official download page; do not use third-party installers.
- Do not choose an Enterprise evaluation license when the free Basic Tier license meets the PYNQ-Z1 need.
- A web installer downloads more components after launch; check disk space before continuing.
- If the user is signing in or accepting terms in a browser, leave those actions to the user unless they explicitly ask to proceed after sign-in.
- Reset and update `STATE.md` following `references/rules/state-rules.md` for each independent run.
