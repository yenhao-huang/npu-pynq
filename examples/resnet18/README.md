# ResNet18 ahead-of-time image demo

Platform guides: [Linux](README.md) | [Windows (PowerShell)](README.win.md) | [macOS](README.macos.md).

This example compiles the pinned TorchVision ResNet-18 checkpoint ahead of time
and classifies a photograph on the PYNQ-Z1 with the 16 x 16 ISA overlay. The
host exports the model through `src/export` into ISA programs, weights and an
ARM shared library; on the board `src/runtime` loads that package and calls
`forward(weights, image, logits)`. The board never imports PyTorch or the
compiler. Run all commands from the repository root in a Linux Bash terminal.

INT8 weights are quantized per output channel and activations per im2col row
inside the compiled graph. This is a different numeric pipeline from the
optional JIT example under `src/jit`.

## Requirements

- Vivado, Python, `ssh` and `scp` are available.
- LLVM/MLIR 22, CMake, clang and the ARM hard-float cross compiler
  (`arm-linux-gnueabihf-gcc`) are installed. Set `NPU_LLVM_BIN` to the LLVM 22
  `bin` directory when it is not found automatically.
- The board is reachable as `192.168.2.99`, user `xilinx`.
- The Git worktree is clean before building deployable artifacts:

```bash
git status --short
```

### Install host dependencies (Ubuntu / WSL Ubuntu)

Run locally from the repository root. Install Vivado separately with Zynq-7000
device support and make `vivado` available on PATH.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip openssh-client tar
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r examples/resnet18/requirements-convert.txt
```

In each new terminal, activate this environment with
`source .venv/bin/activate`. These are host dependencies; the board uses
its own PYNQ Python environment.

Install LLVM/MLIR 22 using the [official Ubuntu packages](https://apt.llvm.org/):

```bash
sudo apt install -y wget ca-certificates gnupg build-essential cmake ninja-build
wget -O /tmp/npu-llvm.sh https://apt.llvm.org/llvm.sh
sudo bash /tmp/npu-llvm.sh 22
sudo apt install -y llvm-22 llvm-22-dev clang-22 lld-22 \
  mlir-22-tools libmlir-22-dev gcc-arm-linux-gnueabihf

export NPU_LLVM_BIN=/usr/lib/llvm-22/bin
export PATH="$NPU_LLVM_BIN:$PATH"
export NPU_BUILD_JOBS=2

"$NPU_LLVM_BIN/mlir-opt" --version
"$NPU_LLVM_BIN/clang" --version
test -f /usr/lib/llvm-22/lib/cmake/mlir/MLIRConfig.cmake
```

Repeat the exports in each new terminal, or add them to `~/.bashrc`.
The exporter builds `npu-opt` automatically on first use.

## 1. Prepare the board once

The notebook needs NumPy, Pillow, matplotlib and ipywidgets in the board's
PYNQ Python environment. Install any that are missing:

```bash
ssh xilinx@192.168.2.99
sudo /usr/local/share/pynq-venv/bin/python3 -m pip install pillow matplotlib ipywidgets
exit
```

## 2. Build the overlay

```bash
vivado -mode batch -nojournal -nolog \
  -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl
```

The default build is the 16 x 16 array with the ISA front end (ABI 1.1) and
publishes BIT/HWH/manifest files under
`build/vivado/npu_matrix_16x16/artifacts/`. An old DMA-only overlay cannot run
this package.

## 3. Build the model

```bash
python -m pip install -r examples/resnet18/requirements-convert.txt
python examples/resnet18/scripts/download_model.py
python examples/resnet18/scripts/download_demo_assets.py
python examples/resnet18/scripts/download_gallery.py
python examples/resnet18/scripts/convert_model.py
```

`convert_model.py` writes the compiled package to `examples/resnet18/model/aot`
for both the host and the board; append `--targets host` to build only host
code. Downloaded weights, images and compiled products stay ignored under
`model/`. Check the package on the host with the simulated NPU:

```bash
python examples/resnet18/run_on_board.py --backend sim \
  --package examples/resnet18/model/aot \
  --image examples/resnet18/model/demo-image.jpg \
  --labels examples/resnet18/model/imagenet-classes.txt \
  --expected-class Samoyed --evidence examples/resnet18/model/host-image.json
```

Host simulation is not physical-board acceptance.

## 4. Package

```bash
commit="$(git rev-parse HEAD)"
tag="local-${commit:0:12}"
mkdir -p dist/resnet18

python examples/resnet18/package_example.py \
  --artifact-dir build/vivado/npu_matrix_16x16/artifacts \
  --report-dir build/vivado/npu_matrix_16x16/reports \
  --output-archive "dist/resnet18/npu-resnet18-$tag.zip" \
  --release-tag $tag --source-commit $commit
```

The archive holds the notebook, the runtime, the compiled model, the labels,
the demo pictures and the verified overlay, with the SHA-256 of every file in
`package.manifest.json`. Packaging fails if the overlay was built from another
commit or its array size differs from the compiled model.

## 5. Deploy and test

Copy the archive to the board and extract it into a notebook directory:

```bash
scp "dist/resnet18/npu-resnet18-$tag.zip" xilinx@192.168.2.99:~/jupyter_notebooks/
ssh xilinx@192.168.2.99 "cd ~/jupyter_notebooks && unzip -o npu-resnet18-$tag.zip -d npu-resnet18"
```

Open <http://192.168.2.99:9090/notebooks/npu-resnet18/resnet18.ipynb> and run
the cells in order:

1. Connect to the package (`NPU_RESNET18_PACKAGE` optionally overrides its path).
2. Program the FPGA (`NPU_RESNET18_OVERLAY`) and load the package.
3. Choose a bundled picture or upload a picture.
4. Preprocess and inspect the original and cropped image.
5. Run the compiled model and inspect NPU counters.
6. Read top-5 labels, correctness for known pictures, and all six block times.

The notebook defaults to the physical `pynq` backend. Block totals exclude
human interaction time between cells. `run_on_board.py` runs the same image
pipeline without the notebook for automated measurements.

## Release CD

`.github/workflows/cd.yml` runs the same steps for each release: it builds the
model and overlay, assembles `npu-resnet18-<tag>.zip` with
`package_example.py`, deploys it with `.github/cd/resnet18_deploy.ps1`, and
classifies the demo picture on the board with
`.github/cd/resnet18_accept_image.ps1`. Both scripts run board commands with
`sudo -n`, so the board needs a reviewed non-interactive sudo policy. Never
store a password or private key in this repository.
