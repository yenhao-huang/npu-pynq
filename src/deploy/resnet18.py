"""Stage a ResNet-18 package exported through PyTorch for the PYNQ-Z1.

    python -m src.deploy.resnet18 dist/resnet18/package --overlay <dir with npu_matrix.bit/.hwh> \
        --assets dist/resnet18/model --out dist/resnet18/board

Adds the example notebook, the ImageNet labels, the demo image and the
gallery (with its licence record) under model/. On the board, from the
bundle directory:

    sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.image_cli \
        --package package --overlay npu_matrix.bit --image model/demo-image.jpg \
        --labels model/imagenet-classes.txt --expected-class Samoyed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.deploy.bundle import REPO, stage

NOTEBOOK = REPO / 'examples' / 'resnet18' / 'resnet18.ipynb'
GALLERY = REPO / 'src' / 'export' / 'model_arch' / 'resnet18' / 'gallery-source.json'


def assets(directory: Path) -> dict[str, Path]:
    files = {NOTEBOOK.name: NOTEBOOK, 'model/gallery-source.json': GALLERY}
    for name in ('imagenet-classes.txt', 'demo-image.jpg'):
        if not (directory / name).is_file():
            raise FileNotFoundError(f'{directory / name}: run python -m src.export.model_arch.resnet18.download demo')
        files[f'model/{name}'] = directory / name
    for entry in json.loads(GALLERY.read_text(encoding='utf-8'))['images']:
        image = directory / entry['filename']
        if image.is_file():
            files[f"model/{entry['filename']}"] = image
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='src.deploy.resnet18')
    parser.add_argument('package', type=Path)
    parser.add_argument('--overlay', type=Path, required=True, help='directory with npu_matrix.bit/.hwh')
    parser.add_argument('--assets', type=Path, default=REPO / 'dist' / 'resnet18' / 'model')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    print(stage(args.package, args.overlay, args.out, assets(args.assets)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
