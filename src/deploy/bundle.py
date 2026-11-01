"""A board bundle: what the PYNQ-Z1 runs for one npu-package/2 package.

    <out>/package/            the package with only the board's CPU library
    <out>/src/...             the runtime and the NPU driver sources (built on the board)
    <out>/npu_matrix.bit/.hwh the overlay
    <out>/<extra files>       notebooks, labels, images

On the board, from the bundle directory, the runtime runs with the PYNQ
environment as root (overlay loading and MMIO), for example:

    sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli run package ...
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
BOARD_CPU = 'arm-cortex-a9'
RUNTIME = ('src/runtime', 'src/vendors/__init__.py', 'src/vendors/npu_pynq/__init__.py',
           'src/vendors/npu_pynq/driver', 'src/vendors/npu_pynq/isa', 'src/vendors/npu_pynq/spec')
SKIP = shutil.ignore_patterns('__pycache__', '*.pyc')


def stage(package: Path, overlay: Path, out: Path, extra: dict[str, Path] | None = None) -> Path:
    """Write the bundle to ``out`` (replaced); ``extra`` maps bundle paths to files."""
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest.get('format') != 'npu-package/2':
        raise ValueError(f'{package}: not an npu-package/2 package')
    if BOARD_CPU not in manifest['libraries']:
        raise ValueError(f'{package} has no {BOARD_CPU} library')
    for suffix in ('bit', 'hwh'):
        if not (overlay / f'npu_matrix.{suffix}').is_file():
            raise FileNotFoundError(overlay / f'npu_matrix.{suffix}')
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
    for rel, src in (extra or {}).items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / rel)
    return out
