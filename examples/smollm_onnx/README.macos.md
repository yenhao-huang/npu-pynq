# Compiled SmolLM2 on macOS

Platform guides: [Linux](README.md) | [Windows (PowerShell)](README.win.md) | [macOS](README.macos.md).

Run local commands from the repository root in the macOS Terminal (Bash or zsh). Commands after an interactive
`ssh` login run in the board's Linux shell. Install Python 3 and make sure
`ssh` and `scp` are available. Keep deployment products under `dist/`.

## Requirements and build boundary

First produce the model package with the [Linux build guide](README.md).
Copy these matching products into this checkout:

- `dist/smollm-package/`: the complete ARM compiled package, including its ONNX Runtime library.
- `build/vivado/npu_matrix_16x16/artifacts/npu_matrix.bit` and `npu_matrix.hwh`: the matching ISA overlay.

Keep the source checkout at the same commit as the package. The board needs
PYNQ, XRT and its existing Python environment.
[AMD lists Windows and Linux as supported Vivado hosts](https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Supported-Operating-Systems). Build the overlay on Linux; this guide deploys the prepared package from macOS without installing Vivado on the Mac.

## Optional ONNX-MLIR import check

The production package still uses the ORT bridge. The new ONNX-MLIR path
inventories and imports the source to MLIR; it does not yet produce an
ARM/NPU package. The pinned compiler uses LLVM 24, lacks an ARM target, and
leaves 151 custom operations requiring conversion. The existing NPU toolchain
uses LLVM 22; interchange remains unverified.

For a local check, use Python 3.10–3.12 and Docker Desktop. The pinned image
targets `linux/amd64`; Apple Silicon requires working amd64 container
emulation. This import path has been tested on Windows with Linux containers,
not on macOS. Keep using the Linux build guide for the production package.
The original ONNX source is required in addition to any compiled package.

Use an existing environment with ONNX 1.17.0, or create one:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install onnx==1.17.0
export NPU_ONNX_MODEL="$PWD/dist/smollm2-onnx"
.venv/bin/python -m src.export.cli prepare-onnx fetch
onnx_mlir_image="$(.venv/bin/python -c 'import json; print(json.load(open("src/export/compilers/onnx_mlir/toolchain.json"))["image"])')"
docker pull --platform linux/amd64 "$onnx_mlir_image"
.venv/bin/python -m src.export.compilers.onnx_mlir.pipeline \
  --source "$NPU_ONNX_MODEL/onnx/model_int8.onnx" \
  --out build/onnx-mlir/smollm-inspect --inventory-only
.venv/bin/python -m src.export.compilers.onnx_mlir.pipeline \
  --source "$NPU_ONNX_MODEL/onnx/model_int8.onnx" \
  --out build/onnx-mlir/smollm-import-check
```

Use new or empty output directories. `inventory.json` records conversion gaps;
successful import produces `model.onnx.mlir`, `compiler.log`, and `import.json`.
`compile_ready` remains false; partition, codegen, and board validation remain
`not-run`. Nothing is dispatched to the FPGA.

See the [import guide](../../docs/onnx-mlir/README.md) and
[integration plan](../../docs/plans/onnx-mlir-incoporate/plan.md). Use the complete
Linux-built package below for deployment and notebook validation.

## 1. Check inputs and deploy

```bash
test -f dist/smollm-package/manifest.json
ssh xilinx@192.168.2.99 "mkdir -p ~/jupyter_notebooks/npu-smollm"
scp -r src dist/smollm-package xilinx@192.168.2.99:~/jupyter_notebooks/npu-smollm/
scp examples/smollm_onnx/smollm_onnx.ipynb xilinx@192.168.2.99:~/jupyter_notebooks/npu-smollm/
scp build/vivado/npu_matrix_16x16/artifacts/npu_matrix.bit build/vivado/npu_matrix_16x16/artifacts/npu_matrix.hwh xilinx@192.168.2.99:~/jupyter_notebooks/npu-smollm/
```

## 2. Run on the board

From Terminal, connect:

```bash
ssh xilinx@192.168.2.99
```

The following commands run in the board's Linux shell. Enter the board sudo
password when prompted:

```bash
cd ~/jupyter_notebooks/npu-smollm
sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli run \
  smollm-package --backend pynq --overlay npu_matrix.bit \
  --chat --stream --prompt "write a paragraph about npu" --max-new 32
```

## 3. Notebook handoff

Open <http://192.168.2.99:9090/notebooks/npu-smollm/smollm_onnx.ipynb> in the
board's root PYNQ kernel. Set `NPU_REPO` to
`/home/xilinx/jupyter_notebooks/npu-smollm`, `NPU_COMPILED_PACKAGE` to its
`smollm-package` directory and `NPU_OVERLAY` to its `npu_matrix.bit` before
running generation. Run CPU simulation in a separate process from physical
inference. Only the compiled prefill lengths are accepted.

## Release CD

See [Release CD](README.md#release-cd) for packaging, sealed hashes and the
separate physical acceptance check. A successful file transfer is not acceptance.
