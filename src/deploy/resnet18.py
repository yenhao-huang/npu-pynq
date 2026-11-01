"""Stage a ResNet-18 package exported through PyTorch for the PYNQ-Z1.

    python -m src.deploy.resnet18 dist/resnet18/package --overlay <dir with npu_matrix.bit/.hwh> \
        --assets dist/resnet18/model --out dist/resnet18/board

Adds the example notebook, the ImageNet labels, the demo image and the
gallery (with its licence record) under model/. On the board, from the
bundle directory:

    sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.image_cli \
        --package package --overlay npu_matrix.bit --image model/demo-image.jpg \
        --labels model/imagenet-classes.txt --expected-class Samoyed

For a release (CD), --release-tag and --source-commit seal the bundle against
the overlay's provenance manifest and --archive writes it as a zip;
``image_cli --package-root`` verifies it on the board. --check-only checks
that a package and its assets are ready, without an overlay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.deploy.bundle import BOARD_CPU, REPO, archive, seal, stage

NOTEBOOK = REPO / 'examples' / 'resnet18' / 'resnet18.ipynb'
GALLERY = REPO / 'src' / 'deploy' / 'resnet18-gallery-source.json'


def assets(directory: Path) -> dict[str, Path]:
    files = {NOTEBOOK.name: NOTEBOOK, 'model/gallery-source.json': GALLERY}
    for name in ('imagenet-classes.txt', 'demo-image.jpg'):
        if not (directory / name).is_file():
            raise FileNotFoundError(f'{directory / name}: run python -m src.deploy.resnet18_assets demo')
        files[f'model/{name}'] = directory / name
    for entry in json.loads(GALLERY.read_text(encoding='utf-8'))['images']:
        image = directory / entry['filename']
        if image.is_file():
            files[f"model/{entry['filename']}"] = image
    return files


def check(package: Path, directory: Path) -> None:
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest.get('format') != 'npu-package/2' or 'classifier' not in manifest:
        raise ValueError(f'{package}: not an image classifier package')
    if BOARD_CPU not in manifest['libraries']:
        raise ValueError(f'{package}: no {BOARD_CPU} library')
    assets(directory)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='src.deploy.resnet18')
    parser.add_argument('package', type=Path)
    parser.add_argument('--overlay', type=Path, help='directory with npu_matrix.bit/.hwh')
    parser.add_argument('--assets', type=Path, default=REPO / 'dist' / 'resnet18' / 'model')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--release-tag')
    parser.add_argument('--source-commit')
    parser.add_argument('--overlay-commit')
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args(argv)
    if args.check_only:
        check(args.package, args.assets)
        print('PASS: the ResNet-18 package and its assets are ready')
        return 0
    if not (args.overlay and args.out):
        parser.error('--overlay and --out are required')
    if bool(args.release_tag) != bool(args.source_commit):
        parser.error('a release needs both --release-tag and --source-commit')
    out = stage(args.package, args.overlay, args.out, assets(args.assets))
    if args.release_tag:
        seal(out, args.release_tag, args.source_commit, args.overlay_commit, args.report_dir)
    if args.archive:
        out = archive(out, args.archive)
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
