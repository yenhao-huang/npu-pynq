# Matrix multiplication on macOS

Platform guides: [Linux](README.md) | [Windows (PowerShell)](README.win.md) | [macOS](README.macos.md).

Use Terminal (Bash or zsh) from the repository root. Local commands use
`ssh` and `scp`; no PowerShell is required.

## Requirements and overlay build

[AMD lists Windows and Linux as supported Vivado hosts](https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Supported-Operating-Systems). Build the overlay and package on a Linux host using
[the Linux guide](README.md#2-build-and-package). Copy the complete generated
package to `dist/matrix-multiplication/package/` on the Mac, preserving
`artifacts/`, `src/`, `runtime/` and the notebook.
The BIT, HWH and manifest must come from the same build.

## 1. Prepare the board once

```bash
ssh -t xilinx@192.168.2.99 'sudo usermod -aG render,video xilinx'
ssh xilinx@192.168.2.99 id
```

Reconnect after changing groups. The output should contain `render` and
`video`. Physical inference also needs the board's MMIO permissions.

## 2. Deploy the package

```bash
test -f dist/matrix-multiplication/package/package.manifest.json
ssh xilinx@192.168.2.99 'mkdir -p ~/jupyter_notebooks/npu-matrix-local'
scp -r dist/matrix-multiplication/package/. xilinx@192.168.2.99:~/jupyter_notebooks/npu-matrix-local/
```

## 3. Board test

Enter the board sudo password when prompted:

```bash
ssh -t xilinx@192.168.2.99 'cd ~/jupyter_notebooks/npu-matrix-local && sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 run_on_board.py --artifact-dir artifacts --release-tag v0.0.0 --evidence board-evidence.json'
mkdir -p dist/matrix-multiplication/evidence
scp xilinx@192.168.2.99:~/jupyter_notebooks/npu-matrix-local/board-evidence.json dist/matrix-multiplication/evidence/
cat dist/matrix-multiplication/evidence/board-evidence.json
```

Expect `PASS: Phase 1C matrix multiplication example`. For a notebook-only
handoff, skip this step; copying files does not program the FPGA.

## 4. Notebook handoff

Open <http://192.168.2.99:9090/notebooks/npu-matrix-local/matrix_multiplication.ipynb>
in the existing PYNQ kernel. Run all cells to verify hashes, program the
FPGA and compare the normal, non-aligned and repeated jobs against NumPy.

## Release CD

See [Release CD](README.md#release-cd) for the separate automated deployment
and acceptance steps. Local notebook execution uses the board's existing
permissions; automated acceptance requires its reviewed non-interactive sudo policy.
