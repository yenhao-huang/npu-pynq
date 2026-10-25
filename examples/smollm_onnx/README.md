# Compiled ONNX SmolLM2 INT8 demo

Platform guides: [Linux](README.md) | [Windows (PowerShell)](README.win.md) | [macOS](README.macos.md).

This example compiles the pinned upstream SmolLM2 INT8 ONNX model ahead of time
and generates text on the PYNQ-Z1 with the 16 x 16 ISA overlay. Integer
contractions run on the NPU; floating-point operators run on the Cortex-A9
through a pinned ARM build of ONNX Runtime 1.20.1. The graph keeps the model's
original quantization. Run all commands from the repository root in Bash.
On Windows, use a WSL Bash terminal with the required tools available there.

## Requirements

- Vivado, Python, Docker, `ssh` and `scp` are available.
- An LLVM/MLIR 22 environment with CMake, clang and the ARM hard-float cross
  compiler (`arm-linux-gnueabihf-gcc`). Set `NPU_LLVM_BIN` to the LLVM 22
  `bin` directory when it is not found automatically.
- The board is reachable as `192.168.2.99`, user `xilinx`, with the PYNQ
  Python environment, XRT and GCC.
- Keep model assets and packages outside `exp/`; the examples below use the
  `dist/` output directory.

### Install host dependencies (Ubuntu / WSL Ubuntu)

Run locally from the repository root. Install Vivado separately with Zynq-7000
device support and make `vivado` available on PATH.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip openssh-client tar
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 onnx==1.17.0 onnxruntime==1.20.1
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

Use Python 3.10–3.12 for these pinned packages (for example, the system Python
in Ubuntu 22.04 or 24.04). Docker must also be installed and running.
Check it before cross-building ONNX Runtime:

```bash
docker version
docker run --rm ubuntu:22.04 uname -m
```

## 1. Build the overlay

```bash
vivado -mode batch -nojournal -nolog \
  -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl
```

The overlay is published under `build/vivado/npu_matrix_16x16/artifacts/`.
Keep `npu_matrix.hwh` next to `npu_matrix.bit` when you copy it.

## 2. Prepare the model

```bash
export NPU_ONNX_MODEL="$PWD/dist/smollm2-onnx"
python -m src.export.cli prepare-onnx prepare
```

`prepare` downloads only the pinned upstream files into `$NPU_ONNX_MODEL`,
checks their SHA-256 values, and exports the native ONNX Runtime operator
sessions. Check generation on the host with the reference interpreter:

```bash
python -m src.runtime.cli reference --backend cpu \
  --prompt "Explain why the sky is blue." --max-new-tokens 24
```

## 3. Build ONNX Runtime for the Cortex-A9

`src/runtime/build_onnxruntime_arm.sh` cross-builds the pinned ONNX Runtime for
Cortex-A9 NEON hard-float in an Ubuntu 22.04 container. The first build can
take hours.

```bash
mkdir -p dist/onnx-runtime-build
out="$PWD/dist/onnx-runtime-build"
script="$PWD/src/runtime/build_onnxruntime_arm.sh"
docker run --rm --mount "type=bind,source=$out,target=/work" \
  --mount "type=bind,source=$script,target=/build.sh,readonly" ubuntu:22.04 bash /build.sh
mkdir -p "$NPU_ONNX_MODEL/onnx-runtime"
cp "$out/libonnxruntime.so.1.20.1" "$out/onnxruntime_c_api.h" "$NPU_ONNX_MODEL/onnx-runtime/"
```

## 4. Compile the package

In the LLVM 22 environment, compile the native float operators and then the
model:

```bash
export NPU_ONNX_MODEL="$PWD/dist/smollm2-onnx"
arm-linux-gnueabihf-gcc -std=c11 -O2 -fPIC \
  -I "$NPU_ONNX_MODEL/onnx-runtime" -c src/runtime/cpu/onnx_ops.c \
  -o dist/onnx_compiled_ops.o
python -m src.export.cli export smollm \
  --source "$NPU_ONNX_MODEL/onnx/model_int8.onnx" \
  --native-object dist/onnx_compiled_ops.o \
  --out dist/smollm-package --prefill-lengths 36,40,42
```

The package holds the ARM library, ISA programs, the original INT8 weights, the
native float operator models and the ARM ONNX Runtime library. ONNX dynamic
quantization scales over the whole prompt, so the runtime accepts only the
compiled chat prompt lengths (`--prefill-lengths`, default 36, 40 and 42).

## 5. Deploy and run

Copy `src/`, the package and the overlay to the board:

```bash
ssh xilinx@192.168.2.99 "mkdir -p ~/jupyter_notebooks/npu-smollm"
scp -r src dist/smollm-package xilinx@192.168.2.99:~/jupyter_notebooks/npu-smollm/
scp build/vivado/npu_matrix_16x16/artifacts/npu_matrix.bit \
  build/vivado/npu_matrix_16x16/artifacts/npu_matrix.hwh \
  xilinx@192.168.2.99:~/jupyter_notebooks/npu-smollm/
```

Then generate on the board, as root because PYNQ needs `/dev/mem` for MMIO:

```bash
ssh xilinx@192.168.2.99
cd ~/jupyter_notebooks/npu-smollm
sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli run \
  smollm-package --backend pynq --overlay npu_matrix.bit \
  --chat --stream --prompt "write a paragraph about npu" --max-new 32
```

`--backend cpu` runs the same ARM library on the simulated NPU; run it in a
separate process from physical inference because the compiled library owns one
NPU runtime state.

## Notebook handoff

[smollm_onnx.ipynb](smollm_onnx.ipynb) imports only `src/`. Open it in the
board's root PYNQ kernel with `NPU_REPO`, `NPU_COMPILED_PACKAGE` and
`NPU_OVERLAY` set to the copied repository, package and overlay. Its generation
cell is editable.

## Release CD

`.github/workflows/cd.yml` runs steps 2 to 4 for each release, seals every
package file with SHA-256 and the release commit, and publishes
`npu-smollm-<tag>.tar.gz`. `.github/cd/smollm_accept.ps1` then generates 32
tokens on the board and requires them to match the frozen reference exactly.
It runs board commands with `sudo -n`, so the board needs a reviewed
non-interactive sudo policy. Never store a password or private key in this
repository. Further detail is in
[docs/goals/smollm-onnx/smollm-onnx.md](../../docs/goals/smollm-onnx/smollm-onnx.md).
