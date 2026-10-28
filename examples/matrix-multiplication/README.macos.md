# Matrix multiplication on PYNQ-Z1 (macOS)

[WSL](README.md) | [macOS](README.macos.md)

Use macOS Terminal with Python 3, NumPy, OpenSSH and tar. Obtain the complete
`dist/matrix-multiplication/local-<commit>` package from the
[WSL build steps](README.md#build-and-package), or extract a matrix package
from a release and set `PACKAGE` to that directory. Build the overlay on a
Windows/Linux Vivado host; this guide deploys an existing package from macOS.
Native macOS execution of this revised guide has not been validated.

## Deploy and run

Run from the repository root. Set `PACKAGE` to the complete directory produced
by the packager. Keep its `src/`, `artifacts/` and notebook together.

```bash
export PACKAGE="dist/matrix-multiplication/local-$(git rev-parse --short=8 HEAD)"
# For an extracted release, set PACKAGE to its actual directory instead.
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
Run every cell in order. The notebook verifies the overlay manifest and uses
`src.runtime.core.matrix.TiledMatrixMultiplier`. `NPU_MATRIX_ARTIFACTS` can
select another verified artifact directory. Notebooks are committed without outputs.

The relocated runtime passed all three cases on the physical 16 x 16 FPGA
(1, 4 and 4 tiles). The complete revised notebook and manifest-verifying CLI
were not rerun: the reused overlay pair has verified hashes but its original
build manifest was unavailable locally. The documented package flow requires
that manifest; no provenance was fabricated. Native macOS execution remains
unverified.
