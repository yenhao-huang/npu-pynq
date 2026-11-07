# SmolLM2 safetensors on PYNQ-Z1

[Linux](README.md) | [Windows](README.win.md) | [macOS](README.macos.md)

On Windows the front end (PyTorch) runs in PowerShell and `pynqo-backend` runs
in WSL Ubuntu; the export command calls WSL itself. The
[Linux guide](README.md) describes the model and the validation result
(PYNQ-Z1, 2026-11-01); this Windows flow produced that package.

## Requirements

- WSL Ubuntu with CMake, Ninja and LLVM/MLIR 22: run the `apt` and LLVM lines
  of the [Linux requirements](README.md#requirements) inside WSL.
- Python 3.12 in PowerShell:

```powershell
py -3.12 -m venv .venv
. ./.venv/Scripts/Activate.ps1
python -m pip install -r src/export/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

- The board and overlay as in the [Linux guide](README.md#requirements).

## Export

```powershell
curl.exe --fail --location --retry 3 --create-dirs `
  --output "examples/smollm-safetensors/model/checkpoint/#1" `
  "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac/{config.json,tokenizer.json,tokenizer_config.json,model.safetensors}"
python -m src.export.cli build src/export/configs/smollm.yml --out dist/smollm/package
```

Set `PYNQO_BACKEND_BUILD` and `PYNQO_MIDDLE_END_BUILD` to WSL paths (for example
`~/npu-build/pynqo-backend` and `~/npu-build/pynqo-middle-end`) to build the tools on
the WSL file system instead of under `build/`.

## Deploy and run

```powershell
$OVERLAY = "build/vivado/npu_matrix_16x16/artifacts"
python -m src.deploy.smollm dist/smollm/package --overlay $OVERLAY --out dist/smollm/board
ssh xilinx@192.168.2.99 "mkdir -p ~/jupyter_notebooks/smollm"
scp -r dist/smollm/board/. xilinx@192.168.2.99:jupyter_notebooks/smollm/
ssh -t xilinx@192.168.2.99 'cd ~/jupyter_notebooks/smollm && sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli run package --overlay npu_matrix.bit --chat --stream --prompt "Write a paragraph about a neural processing unit (NPU)." --max-new 32'
```

Then open <http://192.168.2.99:9090/notebooks/smollm/smollm_safetensors.ipynb>
in the root PYNQ kernel and run all cells.
