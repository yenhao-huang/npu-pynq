# Matrix multiplication on PYNQ-Z1

The smallest way to see the NPU work: multiply matrices on the board and check
every result against NumPy. No model, no compiler, just the bitstream and a
short test.

We last ran the three test cases (one full tile, four edge tiles, repeated
runs) on a 16 x 16 board and they passed. The notebook in this guide hasn't
been rerun since the runtime moved.

## Before you start

You need Linux. On Windows, run `wsl --install -d Ubuntu` in PowerShell, open
Ubuntu, and clone the repository there. You also need Vivado with Zynq-7000
support, and Python 3 with NumPy:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
```

## Build the NPU bitstream and package it

Vivado refuses to build while `src/` has uncommitted changes, so commit those
first. We recommend running Vivado on Windows, where it is installed: run the
same Tcl there, then come back to Linux for the packaging step.

```bash
vivado -mode batch -nojournal -nolog \
  -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl
python -m src.deploy.matrix
```

This checks the bitstream and writes everything the board needs to
`dist/matrix-multiplication/local-<commit>`. Using a bitstream you built
earlier? Add `--artifact-dir <directory>` and tell it which commits it came
from with `--overlay-commit` and `--source-commit`.

## Run on the board

```bash
export PACKAGE="dist/matrix-multiplication/local-$(git rev-parse --short=8 HEAD)"
ssh xilinx@192.168.2.99 'mkdir -p ~/jupyter_notebooks/npu-matrix-local'
scp -r "$PACKAGE/." xilinx@192.168.2.99:jupyter_notebooks/npu-matrix-local/
ssh -t xilinx@192.168.2.99 'cd ~/jupyter_notebooks/npu-matrix-local && sudo XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python3 -m src.test.matrix_acceptance \
  --artifact-dir artifacts --release-tag local-demo --evidence board-evidence.json'
```

It asks for the board's sudo password. You're done when it prints
`PASS: Phase 1C matrix multiplication example`.

Prefer a notebook? Open
<http://192.168.2.99:9090/notebooks/npu-matrix-local/matrix_multiplication.ipynb>,
pick the root PYNQ kernel and run all cells.
