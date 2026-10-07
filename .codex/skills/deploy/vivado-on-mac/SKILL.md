---
name: vivado-on-mac
description: Open the AMD Vivado UI and build the NPU overlay on an Apple silicon Mac (Linux Vivado in Docker under Rosetta, GUI through macOS Screen Sharing), then deploy the overlay and compiled models to the PYNQ-Z1 cabled to the Mac. Use when the user wants Vivado on macOS, the Vivado GUI on the Mac, or a Mac-only build-and-deploy loop.
---

# Vivado on an Apple silicon Mac

Vivado ships only for x86-64 Windows/Linux. On Apple silicon it runs as the
Linux build inside a Docker Desktop container under Rosetta; the container
runs a VNC desktop that macOS Screen Sharing shows. The repository is mounted
at `/work`, and Vivado lives in the Docker volume `npu-vivado-tools`.

```text
vivado_mac.sh check                      engine, amd64 emulation, memory, disk, image, Vivado
vivado_mac.sh build-image                image with Vivado's dependencies + VNC desktop
vivado_mac.sh install-gui INSTALLER.bin  AMD installer in the desktop -> /tools/Xilinx (once)
vivado_mac.sh gui [--fresh]              Vivado UI on build/vivado/npu_matrix_16x16/project/npu_matrix.xpr
vivado_mac.sh build [TCL ARGS]           batch build_overlay.tcl -> BIT/HWH artifacts
vivado_mac.sh deploy [RUN ARGS]          deploy-pynq-macos: overlay --dir <artifacts>, export, run
vivado_mac.sh stop
```

## Workflow

1. `vivado_mac.sh check`. Docker Desktop must run with
   *Use Rosetta for x86_64/amd64 emulation on Apple Silicon* enabled and at
   least 8 GiB of memory (Settings > Resources); synthesis wants 16 GiB.
2. `vivado_mac.sh build-image` (once).
3. The user downloads AMD's *Linux self-extracting web installer* for the
   Vivado version the repository uses (2026.1, matching the self-hosted
   runner) with their own AMD account. Then `vivado_mac.sh install-gui
   <path>.bin` opens the installer in Screen Sharing; the user signs in,
   picks Vivado (ML Standard covers the xc7z020), Zynq-7000 devices only, and
   `/tools/Xilinx`. Never handle the AMD credentials yourself.
4. `vivado_mac.sh gui` creates the project (`--elaborate-only`) if needed and
   opens it in the Vivado UI.
5. `vivado_mac.sh build` produces `build/vivado/npu_matrix_16x16/artifacts`
   (needs a clean Git tree, like CD). `vivado_mac.sh deploy` sends it with the
   packages to the PYNQ-Z1 and returns `build/deploy/results/<id>/results/`.

## Guardrails

- Changing Docker Desktop settings, resetting it, or deleting its data is the
  user's decision; report what `check` found.
- Rosetta emulation is several times slower than native; a full 16 x 16
  implementation can take hours. Timing from this path is real Vivado timing,
  but CD release artifacts still come from the self-hosted runner.
- Never commit project directories or bitstreams.
