# SmolLM2 safetensors on PYNQ-Z1

[Linux](README.md) | [Windows](README.win.md) | [macOS](README.macos.md)

The export back end (`npu-compile`) is built for Linux with LLVM 22, so macOS
cannot export the package yet. Export it on Linux or WSL with the
[Linux guide](README.md) (or the [Windows guide](README.win.md)) and copy
`dist/smollm/board` to the Mac. Deploying from macOS has not been run against
the board; the steps are those of the Linux guide.

## Deploy and run

From the repository root, with OpenSSH:

```bash
ssh xilinx@192.168.2.99 'mkdir -p ~/jupyter_notebooks/smollm'
scp -r dist/smollm/board/. xilinx@192.168.2.99:jupyter_notebooks/smollm/
ssh -t xilinx@192.168.2.99 'cd ~/jupyter_notebooks/smollm && sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli run package --overlay npu_matrix.bit \
  --chat --stream --prompt "Write a paragraph about a neural processing unit (NPU)." --max-new 32'
```

Then open <http://192.168.2.99:9090/notebooks/smollm/smollm_safetensors.ipynb>
in the root PYNQ kernel and run all cells.
