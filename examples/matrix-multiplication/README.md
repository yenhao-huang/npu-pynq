# Matrix multiplication on PYNQ-Z1 (WSL)

[WSL](README.md) | [macOS](README.macos.md)

Run host commands from the repository root in WSL Bash. Install Vivado with
Zynq-7000 support on a supported build host, plus Python 3, NumPy, OpenSSH and
tar. The board uses its existing PYNQ Python and XRT environment.

## Build and package

A deployable overlay requires a clean source checkout. The default build is
16 x 16 with MAX_K=256; arithmetic and physical limits are read from the overlay.
If Vivado is installed on Windows, execute the same Tcl build from its command
shell and make the resulting artifacts accessible to WSL.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
git status --short
vivado -mode batch -nojournal -nolog \
  -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl
python -m src.deploy.matrix
```

The packager verifies the BIT/HWH/manifest and creates
`dist/matrix-multiplication/local-<commit>`. For an existing matching overlay,
pass `--artifact-dir <directory>`; if reusing an earlier build, explicitly pass
its full source commit with `--overlay-commit` and the software commit with
`--source-commit`. Generated artifacts stay outside the example directory.

## Deploy and run

Run from the repository root. Set `PACKAGE` to the complete directory produced
by the packager. Keep its `src/`, `artifacts/` and notebook together.

```bash
export PACKAGE="dist/matrix-multiplication/local-$(git rev-parse --short=8 HEAD)"
export BOARD=xilinx@192.168.2.99
export REMOTE=/home/xilinx/jupyter_notebooks/npu-matrix-local
ssh "$BOARD" "mkdir -p '$REMOTE'"
scp -r "$PACKAGE/." "$BOARD:$REMOTE/"
ssh -t "$BOARD" "cd '$REMOTE' && sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.test.matrix_acceptance --artifact-dir artifacts --release-tag local-demo --evidence board-evidence.json"
```

Success prints `PASS: Phase 1C matrix multiplication example`. The acceptance
covers one full tile, four edge tiles and repeated execution against NumPy.
Enter the board sudo password at the terminal prompt.

Open [matrix_multiplication.ipynb](matrix_multiplication.ipynb) in the board's
root PYNQ kernel at
<http://192.168.2.99:9090/notebooks/npu-matrix-local/matrix_multiplication.ipynb>.
Run every cell in order. The notebook verifies the overlay manifest and tiles
M and N over the physical array itself. `NPU_MATRIX_ARTIFACTS` can
select another verified artifact directory. Notebooks are committed without outputs.

The relocated runtime passed all three cases on the physical 16 x 16 FPGA
(1, 4 and 4 tiles). The complete revised notebook and manifest-verifying CLI
were not rerun: the reused overlay pair has verified hashes but its original
build manifest was unavailable locally. The documented package flow requires
that manifest; no provenance was fabricated. Native macOS execution remains
unverified.
