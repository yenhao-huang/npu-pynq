"""Stage an isolated board demo using byte-identical shared source files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=REPO / 'dist/smollm-safetensors/package')
    parser.add_argument('--checkpoint', type=Path, default=EXAMPLE / 'model/checkpoint')
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=REPO / 'dist/smollm-safetensors/board')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose an unused staging directory: ' + str(args.out))
    manifest = json.loads((args.package / 'manifest.json').read_text())
    assert manifest['meta']['config']['layers'] == 30
    assert manifest['meta']['family'] == 'llama'
    assert manifest['npu']['tasks'] > 0
    assert args.overlay.with_suffix('.hwh').is_file()
    args.out.mkdir(parents=True)
    shutil.copytree(args.package, args.out / 'package', ignore=shutil.ignore_patterns('ir', 'host'))
    shutil.copytree(args.checkpoint, args.out / 'checkpoint')
    for name in ('smollm_safetensors.ipynb', 'model-source.json'):
        shutil.copy2(EXAMPLE / name, args.out / name)
    shutil.copy2(EXAMPLE / 'docs/board-run/data/arm-reference.json', args.out / 'arm-reference.json')
    shutil.copy2(EXAMPLE / 'scripts/execute_notebook.py', args.out / 'execute_notebook.py')
    for directory in ('src/runtime', 'src/vendors/npu_pynq/isa', 'src/vendors/npu_pynq/driver'):
        shutil.copytree(REPO / directory, args.out / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('src/__init__.py', 'src/vendors/__init__.py', 'src/vendors/npu_pynq/__init__.py',
                 'src/export/__init__.py', 'src/export/core/__init__.py',
                 'src/export/core/model_loader.py'):
        source = REPO / name
        if source.is_file():
            dest = args.out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    shutil.copy2(args.overlay, args.out / 'npu_matrix.bit')
    shutil.copy2(args.overlay.with_suffix('.hwh'), args.out / 'npu_matrix.hwh')
    files = {}
    for path in sorted(args.out.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                files[path.relative_to(args.out).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (args.out / 'deployment.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    print('Staged', len(files), 'verified files at', args.out)


if __name__ == '__main__':
    main()
