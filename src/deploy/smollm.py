"""Stage a SmolLM package exported through PyTorch for the PYNQ-Z1.

    python -m src.deploy.smollm dist/smollm/package --overlay <dir with npu_matrix.bit/.hwh> \
        --out dist/smollm/board

Adds the example notebook to the bundle (src/deploy/bundle.py). On the board,
from the bundle directory:

    sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli run package \
        --overlay npu_matrix.bit --chat --stream --prompt "..." --max-new 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.deploy.bundle import REPO, stage

NOTEBOOK = REPO / 'examples' / 'smollm-safetensors' / 'smollm_safetensors.ipynb'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='src.deploy.smollm')
    parser.add_argument('package', type=Path)
    parser.add_argument('--overlay', type=Path, required=True, help='directory with npu_matrix.bit/.hwh')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    print(stage(args.package, args.overlay, args.out, {NOTEBOOK.name: NOTEBOOK}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
