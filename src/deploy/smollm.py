"""Stage a SmolLM package exported through PyTorch for the PYNQ-Z1.

    python -m src.deploy.smollm dist/smollm-pt/package --overlay build/smollm-safetensors \
        --out dist/smollm-pt/board

The bundle holds what the board runs and nothing it does not: the package
(the ARM library, weights, NPU programs, tokenizer), the runtime, the NPU
driver's sources (the board builds libnpu_rt itself on first use), and the
overlay's BIT/HWH pair. On the board, from the bundle directory:

    sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli run package \
        --overlay npu_matrix.bit --chat --stream --prompt "..." --max-new 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
BOARD_CPU = 'arm-cortex-a9'
RUNTIME = ('src/runtime', 'src/vendors/__init__.py', 'src/vendors/npu_pynq/__init__.py',
           'src/vendors/npu_pynq/driver', 'src/vendors/npu_pynq/isa', 'src/vendors/npu_pynq/spec')
SKIP = shutil.ignore_patterns('__pycache__', '*.pyc')


def stage(package: Path, overlay: Path, out: Path) -> Path:
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest.get('format') != 'npu-package/2':
        raise ValueError(f'{package}: not an npu-package/2 package')
    if BOARD_CPU not in manifest['libraries']:
        raise ValueError(f'{package} has no {BOARD_CPU} library')
    if out.exists():
        shutil.rmtree(out)
    dest = out / 'package'
    dest.mkdir(parents=True)
    for item in package.iterdir():
        if item.is_file() and item.name != 'compile.json':
            shutil.copy2(item, dest / item.name)
    library = manifest['libraries'][BOARD_CPU]
    (dest / library).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package / library, dest / library)
    manifest['libraries'] = {BOARD_CPU: library}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    for rel in RUNTIME:
        src = REPO / rel
        if src.is_dir():
            shutil.copytree(src, out / rel, ignore=SKIP)
        else:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / rel)
    for suffix in ('bit', 'hwh'):
        shutil.copy2(overlay / f'npu_matrix.{suffix}', out / f'npu_matrix.{suffix}')
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='src.deploy.smollm')
    parser.add_argument('package', type=Path)
    parser.add_argument('--overlay', type=Path, required=True, help='directory with npu_matrix.bit/.hwh')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    print(stage(args.package, args.overlay, args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
