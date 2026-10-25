# SmolLM2 safetensors on PYNQ-Z1

[Linux](README.md) | [Windows](README.win.md) | [macOS](README.macos.md)

## Validation status

**PASS on PYNQ-Z1, 2026-10-25.** All five notebook code cells completed in a
fresh root kernel. The notebook loaded 272 tensors from the full 30-layer
checkpoint and generated 32 tokens matching the independent ARM ISA reference.
The physical NPU reported 4,114 calls, 1,414,944 jobs and zero ISA errors.
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

Host frontend: Python 3.12+, NumPy, nbformat. Compiler: Linux with LLVM/MLIR 22,
Clang, lld, CMake and a C++ compiler. The existing source does not provide a
native Windows compiler target. macOS frontend staging is supported; use a
Linux compiler host for the documented ARM package workflow.

Board: PYNQ-Z1, PYNQ Python with NumPy, nbformat, nbclient, ipykernel, root kernel,
SSH access, and enough SD space for the original 269 MB checkpoint plus the
136 MB compiled weights. Commands below start at the repository root.
Model assets and generated graph files stay ignored in `model/`; deployment
products stay under `dist/`. No `src` file needs editing or patching.

Create a local Python environment before the commands below:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4 nbformat==5.10.4
```

Install the LLVM/MLIR 22 and CMake build prerequisites described in the
[existing compiler guide](../smollm_onnx/README.md#install-host-dependencies-ubuntu--wsl-ubuntu).
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
python examples/smollm-safetensors/scripts/download_model.py
export NPU_LLVM_BIN=/usr/lib/llvm-22/bin
export NPU_BUILD_JOBS=4
python examples/smollm-safetensors/scripts/convert_model.py build
```

For a split frontend/compiler workflow, use `stage` on a Python 3.12 host,
transfer `examples/smollm-safetensors/model/graph/` and `model/checkpoint/` to
the same paths on the Linux compiler host, then run `compile`. The interchange
contains MLIR, JSON and NumPy arrays; it does not modify shared source.
The compile phase also works with Python 3.11 and NumPy.

## Packaging

```bash
python examples/smollm-safetensors/package_example.py \
  --overlay build/smollm-safetensors/npu_matrix.bit
```

The destination must be unused. The package builder copies shared source byte
for byte, checkpoint, compiled package, notebook and overlay, and seals their
hashes in `deployment.json`. It does not include JIT or the compiler toolchain.

## Deployment and board test

```bash
ssh xilinx@192.168.2.99 'mkdir -p ~/jupyter_notebooks/smollm-safetensors'
scp -r dist/smollm-safetensors/board/. \
  xilinx@192.168.2.99:~/jupyter_notebooks/smollm-safetensors/
ssh -t xilinx@192.168.2.99 \
  'cd ~/jupyter_notebooks/smollm-safetensors && sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 execute_notebook.py'
```

## Notebook handoff

Open `smollm_safetensors.ipynb` in the deployed directory through the board's
Jupyter server. Select the root PYNQ Python kernel and choose **Run All**.
Defaults are relative to the notebook directory; `NPU_SMOLLM_DEMO` can override
it. The notebook verifies every sealed file, reads actual safetensors tensor
values, loads all 30 compiled layers, generates 32 tokens, compares the default
prompt against the ARM ISA reference and requires physical NPU calls/jobs with
zero ISA errors. Prompt and generation length are editable.
The final cell releases CMA buffers.

`execute_notebook.py` runs that same notebook in a fresh kernel and writes
`smollm_safetensors.executed.ipynb`, including failures. `result.json` records
checkpoint identity, generated tokens, timings and NPU counters. Keep the
repository notebook output-free; executed outputs are deployment artifacts.
Generation is greedy. Successful execution proves model loading and physical
execution; response quality is not evaluated by this example.

## Release CD

The existing release CD delivers `examples/smollm_onnx`, with a different pinned
ONNX source and reference output. This safetensors example is a manual workflow
and is not wired into release CD. Its notebook acceptance must be run explicitly;
a passing ONNX release job does not validate this checkpoint path.
