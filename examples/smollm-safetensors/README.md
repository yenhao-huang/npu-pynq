# SmolLM2 safetensors on PYNQ-Z1

[Linux](README.md) | [Windows](README.win.md) | [macOS](README.macos.md)

## Validation status

**Initial validation passed on PYNQ-Z1, 2026-10-25.** All five code cells completed in a
fresh root kernel. The notebook loaded 272 tensors from the full 30-layer
checkpoint and generated 32 tokens matching the independent ARM ISA reference.
The physical NPU reported 4,114 calls, 1,414,944 jobs and zero ISA errors.
The current notebook is simplified to load and run, without deployment hashes
or reference-token checks; it has not been rerun on the board after that edit.
No shared source was changed. See [board evidence and limitations](docs/board-run/README.md).

The validated notebook is deployed at:

```text
/home/xilinx/jupyter_notebooks/smollm-safetensors-20261025/npu_matrix/smollm_safetensors.ipynb
```

## Requirements

Use the current checkout's unmodified `src/`. The input is the pinned Apache-2.0
`HuggingFaceTB/SmolLM2-135M-Instruct` checkpoint, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. All 30 layers are compiled with
W8A8 linear/embedding weights and FP32 KV cache. No ONNX model or JIT interpreter
is used. The host compiles safetensors into an AOT package; the board notebook
reads the original safetensors and runs that package on the physical NPU.

Host frontend: Python 3.12+ and NumPy. Compiler: Linux with LLVM/MLIR 22,
Clang, lld, CMake and a C++ compiler. The existing source does not provide a
native Windows compiler target. Run the source CLI on a Linux compiler host;
Windows and macOS can deploy the resulting ARM package.

Board: PYNQ-Z1, PYNQ Python with NumPy, ipykernel, root kernel,
SSH access, and enough SD space for the original 269 MB checkpoint plus the
136 MB compiled weights. Commands below start at the repository root.
Model assets and generated graph files stay ignored in `model/`; deployment
products stay under `dist/`. No `src` file needs editing or patching.

Create a local Python environment before the commands below:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4
```

Install LLVM/MLIR 22 and the build prerequisites (Ubuntu or WSL Ubuntu). Ubuntu
24.04 does not ship LLVM 22, so it comes from the
[official LLVM packages](https://apt.llvm.org/):

```bash
sudo apt install -y wget ca-certificates gnupg build-essential cmake ninja-build
wget -O /tmp/npu-llvm.sh https://apt.llvm.org/llvm.sh
sudo bash /tmp/npu-llvm.sh 22
sudo apt install -y llvm-22 llvm-22-dev clang-22 lld-22 \
  mlir-22-tools libmlir-22-dev gcc-arm-linux-gnueabihf
```

Verify the toolchain before building:

```bash
/usr/lib/llvm-22/bin/mlir-opt --version
/usr/lib/llvm-22/bin/clang --version
test -f /usr/lib/llvm-22/lib/cmake/mlir/MLIRConfig.cmake
```

Safetensors compilation does not need Docker or ONNX Runtime.

## One-time board preparation

Use the installed PYNQ environment at `/usr/local/share/pynq-venv`. Ensure the
root Jupyter kernel is available. MMIO/CMA require root. Noninteractive acceptance
uses sudo; supply credentials through the existing SSH/sudo mechanism. Never put
passwords in notebook cells or shell arguments.

## Overlay build

Build a 16 x 16 ISA overlay on an authorized Vivado host:

```bash
vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl
```

Use a matching BIT/HWH pair from that build or an existing verified deployment.
This example's physical validation reuses the matching pair from
`~/jupyter_notebooks/smollm-compiled-stage-20261009/` without changing RTL.
The package geometry check rejects a mismatched overlay. Copy the pair to
`build/smollm-safetensors/npu_matrix.bit` and `npu_matrix.hwh` on the host.

## Model build

On Linux with Python 3.12 and LLVM 22 installed:

```bash
curl --fail --location --retry 3 --create-dirs --output "examples/smollm-safetensors/model/checkpoint/#1" "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac/{config.json,tokenizer.json,tokenizer_config.json,model.safetensors}"
export NPU_LLVM_BIN=/usr/lib/llvm-22/bin
export NPU_BUILD_JOBS=4
export NPU_SMOLLM_CHECKPOINT="$PWD/examples/smollm-safetensors/model/checkpoint"
export NPU_COMPILED_PACKAGE="$PWD/dist/smollm-safetensors/package"

python -m src.export.cli export smollm \
  --format safetensors --source "$NPU_SMOLLM_CHECKPOINT" \
  --out "$NPU_COMPILED_PACKAGE" --targets host,pynq \
  --bits 8 --embed-bits 8 --kv f32 --attention online \
  --chunk 16 --max-seq 512

cp "$NPU_SMOLLM_CHECKPOINT/tokenizer_config.json" "$NPU_COMPILED_PACKAGE/"
```

The single curl command downloads all four files at the pinned revision.
Pinned file hashes are recorded in `model-source.json` for reference.
The model is loaded, quantized and compiled directly by `src.export.cli`; no example-local
Python build script or staged graph is required. Omitting `--layers` keeps all
30 layers. The source frontend bundles `tokenizer.json`; copy the original
`tokenizer_config.json` as shown so the runtime also applies the model's chat
template. This copies a model asset without changing `src/`.

## Packaging

Use a new deployment directory and copy the runtime, model, notebook and overlay
in nine lines. Shared source files are copied unchanged.

```bash
board=dist/smollm-safetensors/board
mkdir -p "$board/src/vendors/npu_pynq" "$board/src/export/core"
cp -R dist/smollm-safetensors/package "$board/"
cp -R examples/smollm-safetensors/model/checkpoint "$board/"
cp examples/smollm-safetensors/smollm_safetensors.ipynb "$board/"
cp -R src/runtime "$board/src/"
cp -R src/vendors/npu_pynq/isa src/vendors/npu_pynq/driver "$board/src/vendors/npu_pynq/"
cp src/export/core/model_loader.py "$board/src/export/core/"
cp build/smollm-safetensors/npu_matrix.bit build/smollm-safetensors/npu_matrix.hwh "$board/"
```

## Deployment and board test

```bash
ssh xilinx@192.168.2.99 'mkdir -p ~/jupyter_notebooks/smollm-safetensors'
scp -r dist/smollm-safetensors/board/. \
  xilinx@192.168.2.99:~/jupyter_notebooks/smollm-safetensors/
ssh -t xilinx@192.168.2.99 \
  'cd ~/jupyter_notebooks/smollm-safetensors && sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli run package --backend pynq --overlay npu_matrix.bit --chat --stream --prompt "Write a paragraph about a neural processing unit (NPU)." --max-new 32'
```

## Notebook handoff

Open `smollm_safetensors.ipynb` in the deployed directory through the board's
Jupyter server. Select the root PYNQ Python kernel and choose **Run All**.
Defaults are relative to the notebook directory; `NPU_SMOLLM_DEMO` can override
it. The notebook loads the original safetensors, opens the compiled full model
and physical overlay, then streams a generated response and prints runtime
statistics. Edit the prompt and token limit in the generation cell. The final
cell releases CMA buffers. Keep the repository notebook output-free.

## Release CD

The existing release CD delivers `examples/smollm_onnx`, with a different pinned
ONNX source and reference output. This safetensors example is a manual workflow
and is not wired into release CD. Its notebook acceptance must be run explicitly;
a passing ONNX release job does not validate this checkpoint path.
