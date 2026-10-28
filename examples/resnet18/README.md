# ResNet-18 on PYNQ-Z1 (WSL)

[WSL](README.md) | [macOS](README.macos.md)

Run host commands from the repository root in WSL Bash. The exporter compiles
CPU operators to an ARM shared library and INT8 contractions to NPU programs;
the board loads the compiled package without PyTorch or the compiler.

## Prepare the host

Install Vivado with Zynq-7000 support on a Windows/Linux build host and
LLVM/MLIR 22 development tools in WSL. Use the [LLVM packages](https://apt.llvm.org/)
to make the versioned packages available before running:

```bash
sudo apt update
sudo apt install -y python3-venv openssh-client build-essential cmake ninja-build \
  llvm-22 llvm-22-dev clang-22 lld-22 mlir-22-tools libmlir-22-dev \
  gcc-arm-linux-gnueabihf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r src/export/graphs/resnet18/requirements.txt
export NPU_LLVM_BIN=/usr/lib/llvm-22/bin
export PATH="$NPU_LLVM_BIN:$PATH"
export NPU_BUILD_JOBS=2
```

## Compile and package

Build a deployable overlay from a clean source checkout. The default 16 x 16
ISA overlay publishes matching BIT/HWH/manifest files in
`build/vivado/npu_matrix_16x16/artifacts`. If Vivado is on Windows, run the Tcl
build there and make its outputs accessible to WSL. The exporter builds
`npu-opt` automatically on first use.

```bash
git status --short
vivado -mode batch -nojournal -nolog \
  -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl
python -m src.export.graphs.resnet18.download_model
python -m src.export.graphs.resnet18.download_demo_assets
python -m src.export.graphs.resnet18.download_gallery
python -m src.export.cli export resnet18 \
  --source dist/resnet18/model/resnet18-f37072fd.pth \
  --out dist/resnet18/model/aot
python -m src.runtime.image_cli --backend sim \
  --package dist/resnet18/model/aot --image dist/resnet18/model/demo-image.jpg \
  --labels dist/resnet18/model/imagenet-classes.txt --expected-class Samoyed
commit="$(git rev-parse HEAD)"
python -m src.export.delivery.resnet18 \
  --artifact-dir build/vivado/npu_matrix_16x16/artifacts \
  --output-archive dist/resnet18/npu-resnet18.zip \
  --release-tag "local-${commit:0:12}" --source-commit "$commit"
```

Downloaded checkpoints, gallery images and compiled products stay under
`dist/resnet18/model`; pinned source metadata lives in `src/export/graphs/resnet18`.
Downloaders verify size and SHA-256 and refuse to overwrite differing files.
Packaging checks model/overlay geometry and source provenance, and hashes every
file in the delivery. For a reused overlay, pass its full build commit as
`--overlay-commit`.

## Deploy and run

Use the archive produced below or a release archive containing the AOT model,
matching ISA ABI 1.1 overlay and delivery manifest. An older JIT package is not
interchangeable. Run these commands locally from the repository root:

```bash
export BOARD=xilinx@192.168.2.99
export REMOTE=/home/xilinx/jupyter_notebooks/npu-resnet18
scp dist/resnet18/npu-resnet18.zip "$BOARD:~/jupyter_notebooks/"
ssh "$BOARD" "mkdir -p '$REMOTE' && /usr/local/share/pynq-venv/bin/python3 -m zipfile -e /home/xilinx/jupyter_notebooks/npu-resnet18.zip '$REMOTE'"
ssh -t "$BOARD" "cd '$REMOTE' && sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.image_cli --package-root . --expected-class Samoyed --evidence image-acceptance.json"
```

Enter the board sudo password when prompted. Success reports Samoyed as top-1
and a positive physical NPU job count. Install missing notebook dependencies
in the board's PYNQ environment if necessary:

```bash
ssh -t "$BOARD" 'sudo /usr/local/share/pynq-venv/bin/python3 -m pip install pillow matplotlib ipywidgets'
```

Open [resnet18.ipynb](resnet18.ipynb) in the root PYNQ kernel at
<http://192.168.2.99:9090/notebooks/npu-resnet18/resnet18.ipynb>.
Run the six stages in order: connect, load, choose/upload, preprocess, infer,
and inspect top-5 predictions. The gallery retains image licensing metadata.
`NPU_RESNET18_PACKAGE` and `NPU_RESNET18_OVERLAY` override package and overlay
paths; `NPU_RESNET18_BACKEND=sim` explicitly selects host simulation.

The revised six-stage notebook and image CLI passed on PYNQ-Z1 with an existing
16 x 16 overlay and compiled model: top-1 Samoyed (89.80%), 35,174 physical NPU
jobs. Gallery downloads and interactive uploads were not exercised in that run.
A fresh Vivado build and the full release workflow were not rerun. Notebooks are
committed without outputs.
