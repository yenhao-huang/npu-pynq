# SmolLM2 ONNX on PYNQ-Z1 (macOS)

[WSL](README.md) | [macOS](README.macos.md)

Run host commands from the repository root in macOS Terminal. Obtain
`dist/smollm-package/` from the [WSL compilation steps](README.md#1-fetch-and-compile)
and supply the matching 16×16, MAX_K=256, ISA ABI 1.1 overlay pair. Keep the source
checkout consistent with the package. The package compiles CPU operators to
ARM32 and NPU contractions to ISA programs; the board does not need ONNX Runtime.

This guide covers deployment from macOS. The compiler and physical acceptance
for this revision were exercised from the Windows/Linux toolchain and PYNQ-Z1.

## 1. Deploy

```bash
export NPU_BIT="$PWD/build/vivado/npu_matrix_16x16/artifacts/npu_matrix.bit"
export NPU_HWH="$PWD/build/vivado/npu_matrix_16x16/artifacts/npu_matrix.hwh"
test -f dist/smollm-package/manifest.json
export BOARD=xilinx@192.168.2.99
export REMOTE=/home/xilinx/jupyter_notebooks/npu-smollm
ssh "$BOARD" "mkdir -p '$REMOTE'"
tar --exclude='__pycache__' -cf - src/runtime src/vendors/__init__.py \
  src/vendors/npu_pynq/__init__.py src/vendors/npu_pynq/isa src/vendors/npu_pynq/driver \
  | ssh "$BOARD" "tar -xf - -C '$REMOTE'"
scp -r dist/smollm-package "$BOARD:$REMOTE/"
scp examples/smollm_onnx/smollm_onnx.ipynb "$BOARD:$REMOTE/"
scp "$NPU_BIT" "$BOARD:$REMOTE/npu_matrix.bit"
scp "$NPU_HWH" "$BOARD:$REMOTE/npu_matrix.hwh"
```

## 2. Generate on the board

These commands run in the board's Linux shell. Enter the sudo password when
prompted. The board uses its existing PYNQ Python environment and XRT.

```bash
ssh xilinx@192.168.2.99
cd /home/xilinx/jupyter_notebooks/npu-smollm
sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli run \
  smollm-package --backend pynq --overlay npu_matrix.bit --chat --stream \
  --prompt "Write a paragraph about a neural processing unit (NPU)." --max-new 32
```

## 3. Run the notebook

Open [smollm_onnx.ipynb](smollm_onnx.ipynb) in the root PYNQ kernel at
<http://192.168.2.99:9090/notebooks/npu-smollm/smollm_onnx.ipynb>.
Run setup, generation and cleanup in order. Defaults are the notebook directory,
`smollm-package/` and `npu_matrix.bit`. Override `NPU_REPO`,
`NPU_COMPILED_PACKAGE` and `NPU_OVERLAY` in the kernel for another layout.
The complete chat input length must be compiled; defaults are 36, 40 and 42.

Physical acceptance passed all three prefill profiles and three decode steps
each, with exact KV state and greedy tokens and maximum logit error `4.2e-5`.
The [integration guide](../../docs/plans/onnx-mlir-incoporate/README.md) records
the numerical and hardware evidence. The notebook is committed without outputs.
