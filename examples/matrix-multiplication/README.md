# Matrix multiplication deployment

Platform guides: [Linux](README.md) | [Windows (PowerShell)](README.win.md) | [macOS](README.macos.md).

This example defaults to a 16 x 16 systolic array. It builds the NPU overlay,
deploys it to PYNQ-Z1, runs three matrix tests, and downloads JSON evidence. Run all Linux Bash commands from the
repository root.

## Requirements

- Vivado, Python, `ssh`, `scp`, and `tar` are available.
- The board is reachable as `pynq` or `192.168.2.99`.
- The Git worktree is clean before building deployable artifacts:

```bash
git status --short
```

The command must print nothing. A dirty exploratory build may use
`-tclargs --allow-dirty`, but it deliberately does not publish deployable
artifacts.

### Install host dependencies (Ubuntu / WSL Ubuntu)

Run locally from the repository root. Install Vivado separately with Zynq-7000
device support and make `vivado` available on PATH.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip openssh-client tar
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
```

In each new terminal, activate this environment with
`source .venv/bin/activate`. These are host dependencies; the board uses
its own PYNQ Python environment.

## 1. Prepare the board once

```bash
ssh pynq
sudo usermod -aG render,video xilinx
exit
ssh pynq id
```

The final output should include `render` and `video`. PYNQ also requires root
for `/dev/mem` MMIO, so local deployment will ask for the board sudo password.

## 2. Build and package

```bash
vivado -mode batch -nojournal -nolog \
  -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl

python examples/matrix-multiplication/package_example.py
```

The package is created under
`dist/matrix-multiplication/local-<commit>`.

## 3. Deploy and test

The default package path contains the source commit. Copy the complete package
without flattening its directories:

```bash
commit="$(git rev-parse HEAD)"
package="dist/matrix-multiplication/local-${commit:0:8}"
test -d "$package"
ssh xilinx@192.168.2.99 "mkdir -p ~/jupyter_notebooks/npu-matrix-local"
scp -r "$package/." xilinx@192.168.2.99:~/jupyter_notebooks/npu-matrix-local/
```

Run the acceptance separately from the transfer. Enter the board sudo password
when prompted:

```bash
ssh -t xilinx@192.168.2.99 'cd ~/jupyter_notebooks/npu-matrix-local && sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 run_on_board.py --artifact-dir artifacts --release-tag v0.0.0 --evidence board-evidence.json'
mkdir -p dist/matrix-multiplication/evidence
scp xilinx@192.168.2.99:~/jupyter_notebooks/npu-matrix-local/board-evidence.json dist/matrix-multiplication/evidence/
cat dist/matrix-multiplication/evidence/board-evidence.json
```

Success prints `PASS: Phase 1C matrix multiplication example`. For notebook-only
deployment, skip the acceptance command and continue to the notebook below.

## Release CD

`.github/workflows/cd.yml` uses the PowerShell deployment wrapper in deploy-only
mode and a separate acceptance step. Automated acceptance uses `sudo -n`;
the dedicated board must have
a reviewed non-interactive sudo policy before GitHub Actions CD can pass. Never
store a password or private key in this repository.

## 16 x 16 notebook handoff

The default build publishes matching BIT/HWH/manifest files under
`build/vivado/npu_matrix_16x16/artifacts/`. Explicit `--array-size 2` builds
remain under `build/vivado/npu_matrix/`; pass that artifact directory to
`package_example.py --artifact-dir` when packaging a 2 x 2 demo.

To leave execution to the notebook operator, copy the complete generated
package to a dedicated directory under `/home/xilinx/jupyter_notebooks/`.
Keep `artifacts/`, `runtime/`, and `src/` alongside
`matrix_multiplication.ipynb`. Do not run the automated deploy-and-test
command in section 3 for this handoff.

Open `matrix_multiplication.ipynb` in the board's existing PYNQ Python kernel.
The notebook verifies artifact hashes before loading the overlay and prints
the physical limits; the default is `(16, 16, 256)`. It checks an 8 x 8
output tile, a 9 x 9 output tile, and a repeated job
against NumPy. Notebook execution programs the FPGA. Use a PYNQ kernel with
the board's existing MMIO/DMA permissions.
