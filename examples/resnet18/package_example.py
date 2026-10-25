"""Bundle compiled ResNet18 and the independent runtime for board delivery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR if (MODULE_DIR/'src/runtime').is_dir() else MODULE_DIR.parents[1]
sys.path.insert(0, str(ROOT))
from src.vendors.npu_pynq.driver.overlay.verify import verify_artifacts


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(model_dir):
    model_dir = Path(model_dir)
    package = model_dir / "aot"
    manifest = json.loads((package / "manifest.json").read_text())
    if manifest.get("format") != "npu-package/1" or manifest["meta"].get("family") != "resnet18":
        raise ValueError("Expected a compiled ResNet18 package")
    for name in ("weights.bin", "programs.json", "programs.bin", *manifest["libraries"].values()):
        path = (package / name).resolve()
        if not path.is_relative_to(package.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe compiled asset: {name}")
    if "pynq" not in manifest["libraries"]:
        raise ValueError("A board release requires an ARM library")
    for name in ("imagenet-classes.txt", "demo-image.jpg"):
        if not (model_dir / name).is_file():
            raise ValueError(f"Missing demo asset: {name}")
    return manifest


def verify_delivery(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'package.manifest.json').read_text())
    if manifest.get('format') != 'npu-resnet18-aot/1':
        raise ValueError('Expected a ResNet18 AOT delivery')
    for name, expected in manifest['files'].items():
        candidate = root/name
        path = candidate.resolve()
        if candidate.is_symlink() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or unsafe delivery file: '+name)
        if sha256_file(path) != expected:
            raise ValueError('Delivery hash mismatch: '+name)
    return manifest


def build_archive(repo, model_dir, artifacts, output, release_tag, source_commit, overlay_commit=None, report_dir=None):
    repo, model_dir, artifacts, output = map(Path, (repo, model_dir, artifacts, output))
    compiled = validate_model(model_dir)
    overlay = verify_artifacts(artifacts)
    limits = compiled.get('limits', {})
    if limits.get('rows') != overlay['array_size'] or limits.get('columns') != overlay['array_size']:
        raise ValueError('Compiled package geometry differs from the overlay')
    if overlay["source_commit"] != (overlay_commit or source_commit):
        raise ValueError("Overlay source commit differs from the declared build")
    files = {}
    def add_tree(source, destination, suffixes=None):
        for p in source.rglob("*"):
            if p.is_file() and '__pycache__' not in p.parts and (suffixes is None or p.suffix in suffixes):
                if p.is_symlink():
                    raise ValueError(f"Symlink in release inputs: {p}")
                files[str(Path(destination) / p.relative_to(source)).replace('\\', '/')] = p
    add_tree(repo / 'src/runtime', 'src/runtime', {'.py'})
    # The board needs the NPU's ISA and driver, never its compiler backend.
    for name in ('src/vendors/__init__.py', 'src/vendors/npu_pynq/__init__.py'):
        files[name] = repo / name
    for part in ('isa', 'driver'):
        add_tree(repo / 'src/vendors/npu_pynq' / part, f'src/vendors/npu_pynq/{part}', {'.py'})
    add_tree(model_dir / 'aot', 'model/aot')
    for name in ('imagenet-classes.txt', 'demo-image.jpg'):
        files['model/'+name] = model_dir/name
    example = repo / 'examples/resnet18'
    gallery_path = example/'gallery-source.json'
    gallery = json.loads(gallery_path.read_text())
    for entry in gallery['images']:
        name = entry['filename']
        path = (model_dir/name).resolve()
        if not path.is_relative_to(model_dir.resolve()):
            raise ValueError('Unsafe gallery filename')
        if path.is_file():
            files['model/'+name] = path
    for name in ('resnet18.ipynb', 'run_on_board.py', 'accept_image_on_board.py',
                 'package_example.py', 'gallery-source.json', 'README.md'):
        files[name] = example/name
    for name in ('npu_matrix.bit', 'npu_matrix.hwh', 'npu_matrix.manifest.json'):
        files['artifacts/'+name] = artifacts/name
    if report_dir:
        add_tree(Path(report_dir), 'reports')
    manifest = {'format':'npu-resnet18-aot/1', 'release_tag':release_tag,
                'source_commit':source_commit, 'overlay_commit':overlay_commit or source_commit,
                'files':{name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sorted(files.items())}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, p in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, p.read_bytes())
        archive.writestr(zipfile.ZipInfo('package.manifest.json', (1980,1,1,0,0,0)),
                         json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    return output


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--repository-root', type=Path, default=ROOT)
    p.add_argument('--model-dir', type=Path, default=ROOT/'examples/resnet18/model')
    p.add_argument('--artifact-dir', type=Path)
    p.add_argument('--report-dir', type=Path)
    p.add_argument('--output-archive', type=Path)
    p.add_argument('--release-tag')
    p.add_argument('--source-commit')
    p.add_argument('--overlay-commit')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args(argv)
    if a.check_only:
        validate_model(a.model_dir)
        print('PASS: AOT board model workspace is complete')
        return 0
    if not all((a.artifact_dir, a.output_archive, a.release_tag, a.source_commit)):
        p.error('Release assembly requires --artifact-dir, --output-archive, --release-tag and --source-commit')
    print(build_archive(a.repository_root,a.model_dir,a.artifact_dir,a.output_archive,
                        a.release_tag,a.source_commit,a.overlay_commit,a.report_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
