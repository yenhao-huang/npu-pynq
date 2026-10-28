# SmolLM2 ONNX on PYNQ-Z1 (WSL)

[WSL](README.md) | [macOS](README.macos.md)

Run host commands from the repository root in WSL Bash. ONNX-MLIR imports the
pinned INT8 model before CPU/NPU partition. LLVM 22 compiles CPU operators to
ARM32; the existing NPU compiler emits ISA programs and packs INT8 weights.
The board runs the compiled package without ONNX Runtime or a bridge object.

## Requirements

Use Python 3.10–3.12, Docker with Linux containers, CMake 3.24+, Ninja,
LLVM/MLIR 22 development libraries, clang 22 and lld. Set `NPU_LLVM_BIN` to
the LLVM 22 `bin` directory. The exporter builds its parser adapter and
`npu-opt` on first use. Compiler details are in the
[integration guide](../../docs/plans/onnx-mlir-incoporate/README.md).

The board needs its existing PYNQ Python environment and XRT. Supply a matching
16×16, MAX_K=256, ISA ABI 1.1 BIT/HWH pair. Reuse a verified pair or build it on
your Vivado host with `src/hw/vivado_tcl/npu_matrix/build_overlay.tcl`.

## 1. Fetch and compile

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4 onnx==1.17.0
export NPU_LLVM_BIN=/usr/lib/llvm-22/bin
export CMAKE_GENERATOR=Ninja
export NPU_BUILD_JOBS=2
export NPU_ONNX_MODEL="$PWD/dist/smollm2-onnx"
docker pull ghcr.io/onnxmlir/onnx-mlir@sha256:5df1c1d3f1d8d5207c9b408e9d6e0ec2ae9926c6948b4b5c745358a2b47ad2e8
python -m src.export.cli prepare-onnx fetch
python -m src.export.cli export smollm \
  --source "$NPU_ONNX_MODEL/onnx/model_int8.onnx" \
  --out dist/smollm-package --prefill-lengths 36,40,42
```

Imports and compiler builds are cached under ignored `build/`. To reuse a
separately prepared import, pass `--onnx-import-dir <directory>`.
The output contains ARM/host libraries, weights, ISA programs and tokenizer
assets. Prefill lengths count complete chat input tokens. Dynamic quantization
uses the whole prompt, so runtime input must match a compiled specialization.

`prepare-onnx prepare` additionally exports and verifies a reference JSON/tensor
graph; it does not create MLIR. The `export` command performs ONNX-MLIR import.
`prepare-onnx operators` separately prepares optional host ORT references.

## 2. Deploy

Set these paths to your matching overlay pair:

```bash
export NPU_BIT="$PWD/build/vivado/npu_matrix_16x16/artifacts/npu_matrix.bit"
export NPU_HWH="$PWD/build/vivado/npu_matrix_16x16/artifacts/npu_matrix.hwh"
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

## 3. Generate on the board

The following commands run in the board's shell. Enter the sudo password when
prompted; root access is required for overlay loading and MMIO.

```bash
ssh xilinx@192.168.2.99
cd /home/xilinx/jupyter_notebooks/npu-smollm
sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli run \
  smollm-package --backend pynq --overlay npu_matrix.bit --chat --stream \
  --prompt "Write a paragraph about a neural processing unit (NPU)." --max-new 32
```

## 4. Run the notebook

Open [smollm_onnx.ipynb](smollm_onnx.ipynb) in the board's root PYNQ kernel at
<http://192.168.2.99:9090/notebooks/npu-smollm/smollm_onnx.ipynb>.
Run setup, generation and cleanup in order. Defaults match the deployment above.
For another layout, set `NPU_REPO`, `NPU_COMPILED_PACKAGE` and `NPU_OVERLAY`
in that kernel before setup. Edit prompts within the compiled chat lengths.

Physical acceptance passed prefill 36/40/42 and three decode steps each:
KV state and greedy tokens matched the independent source ONNX reference;
maximum logit error was `4.2e-5`. See the integration guide for evidence and
reproduction commands. The notebook is committed without outputs.
