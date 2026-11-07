# SmolLM2 on PYNQ-Z1

Export SmolLM2-135M-Instruct on your computer, then chat with it on the board.
The NPU handles the matrix multiplications, the ARM CPU does the rest.

We last ran this on the board on 2026-11-01: the output matched the host
simulator. Don't expect speed: prefill takes about 48 s and decode about
0.5 tokens/s.

## Before you start

You need Linux. On Windows, run `wsl --install -d Ubuntu` in PowerShell, open
Ubuntu, and clone the repository there (under `~`, not `/mnt/c`, or the first
build crawls).

Install the tools, from the repository root:

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

## Export

```bash
curl --fail --location --retry 3 --create-dirs \
  --output "examples/smollm-safetensors/model/checkpoint/#1" \
  "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac/{config.json,tokenizer.json,tokenizer_config.json,model.safetensors}"
python -m src.export.cli build src/export/configs/smollm.yml --out dist/smollm/package
```

The first time, this builds the compiler tools: go get a coffee, it takes
about 15 minutes and 4 GB of memory. After that an export takes about five.

## Build the NPU bitstream (recommended on Windows)

The export targets a 16 x 16 NPU, so the bitstream must be one too. Build it
on Windows, where Vivado is installed; PowerShell from the repository root
works. It refuses to run while `src/` has uncommitted changes.

```powershell
vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl -tclargs --out dist/smollm/overlay
```

Already have a 16 x 16 `npu_matrix.bit` and `npu_matrix.hwh`? Skip Vivado and
put them in `dist/smollm/overlay/`. Cloned the repository inside WSL? Copy
the two files from Windows into WSL's `dist/smollm/overlay/`.

## Run on the board

### Connect

Plug the board into your computer with an Ethernet cable and power it on.
The board is at `192.168.2.99`, so give your computer's Ethernet port an
address on the same network, such as `192.168.2.1`, with mask
`255.255.255.0`. Then check you can log in (it asks for the board's
password):

```bash
ssh xilinx@192.168.2.99 'echo connected'
```

Tired of typing the password? Run `ssh-copy-id xilinx@192.168.2.99` once.

### Send

```bash
src/deploy/deploy.sh dist/smollm examples/smollm-safetensors/smollm_safetensors.ipynb
```

This copies the runtime and the NPU driver into `dist/smollm/src`, adds the
notebook, and sends the whole `dist/smollm` (package, overlay, runtime) to
`~/jupyter_notebooks/smollm` on the board with `scp`. Your board somewhere
else? Set `BOARD`, for example `BOARD=xilinx@10.0.0.5 src/deploy/deploy.sh ...`.

### Chat

```bash
ssh -t xilinx@192.168.2.99 'cd ~/jupyter_notebooks/smollm && sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli llm package --overlay overlay/npu_matrix.bit \
  --chat --stream --prompt "Write a paragraph about a neural processing unit (NPU)." --max-new 32'
```

It asks for the board's sudo password. The first run spends about 30 s
compiling the board-side libraries.

Prefer a notebook? Open
<http://192.168.2.99:9090/notebooks/smollm/smollm_safetensors.ipynb>, pick the
root PYNQ kernel and run all cells.
