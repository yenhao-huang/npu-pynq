# SmolLM2 safetensors on PYNQ-Z1

[Linux](README.md) | [Windows](README.win.md) | [macOS](README.macos.md)

The host exports `HuggingFaceTB/SmolLM2-135M-Instruct` through PyTorch:
transformers loads it, our torchao config quantizes every linear layer to
W8A8, `torch.export` captures decode and prefill, torch-mlir imports them,
`npu-middle-end` lowers them to linalg, and `npu-compile` (LLVM 22) builds the
package. On the board the CPU
code runs on the Cortex-A9 and every INT8 contraction runs on the NPU.

**Validated on PYNQ-Z1, 2026-11-01**, with the commands below and the notebook
(fresh root kernel, all cells): 32 tokens, "A neural processing unit (NPU) is
a type of computer component that processes information in a way that mimics
the way the human brain processes information. It is", identical to the host
simulator; 7,174 NPU calls, 1,414,944 jobs, no ISA error. Prefill takes about
48 s and decode about 0.5 tokens/s.

## Requirements

Ubuntu or WSL Ubuntu with Python 3.12, CMake, Ninja and LLVM/MLIR 22 from the
[LLVM packages](https://apt.llvm.org/). Commands start at the repository root.

```bash
sudo apt install -y python3-venv build-essential cmake ninja-build wget
wget -O /tmp/npu-llvm.sh https://apt.llvm.org/llvm.sh
sudo bash /tmp/npu-llvm.sh 22
sudo apt install -y llvm-22-dev clang-22 lld-22 mlir-22-tools libmlir-22-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/export/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

The board needs its PYNQ environment, g++ and CMake (the PYNQ image has them),
and a 16 x 16, MAX_K=256, ISA ABI 1.1 BIT/HWH pair. Build one with
`vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl`
or reuse a verified pair.

## Export

```bash
curl --fail --location --retry 3 --create-dirs \
  --output "examples/smollm-safetensors/model/checkpoint/#1" \
  "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac/{config.json,tokenizer.json,tokenizer_config.json,model.safetensors}"
python -m src.export.cli build src/export/configs/smollm.yml --out dist/smollm/package
```

The checkpoint's hashes are pinned in `model-source.json`. The first build
downloads torch-mlir and compiles `npu-middle-end` and `npu-compile` under
`build/`, which takes about 15 minutes on two cores and needs 4 GB of memory;
the export itself takes about five minutes.
`src/export/configs/smollm.yml` holds the model source, the quantization, the
cache length (256) and the prefill chunk (16).

## Deploy and run

Set `OVERLAY` to the directory holding `npu_matrix.bit` and `npu_matrix.hwh`:

```bash
export OVERLAY=build/vivado/npu_matrix_16x16/artifacts
python -m src.deploy.smollm dist/smollm/package --overlay "$OVERLAY" --out dist/smollm/board
ssh xilinx@192.168.2.99 'mkdir -p ~/jupyter_notebooks/smollm'
scp -r dist/smollm/board/. xilinx@192.168.2.99:jupyter_notebooks/smollm/
ssh -t xilinx@192.168.2.99 'cd ~/jupyter_notebooks/smollm && sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli llm package --overlay npu_matrix.bit \
  --chat --stream --prompt "Write a paragraph about a neural processing unit (NPU)." --max-new 32'
```

The first run builds libnpu_rt and the delegate dispatcher on the board (about
30 s). Enter the board's sudo password when asked.

## Notebook

Open <http://192.168.2.99:9090/notebooks/smollm/smollm_safetensors.ipynb>,
select the root PYNQ Python kernel and run all cells. Edit the prompt in the
generation cell. The last cell releases the board buffers.
