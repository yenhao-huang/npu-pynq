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
