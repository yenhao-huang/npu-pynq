"""Create an allowlisted board payload without JIT, PyTorch or the exporter."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[4]


def main():
    files = {}
    for folder in ('src/runtime','src/isa'):
        for p in (ROOT/folder).rglob('*.py'):
            files[p.relative_to(ROOT).as_posix()] = p
    example = ROOT/'examples/resnet18'
    for name in ('resnet18.ipynb','run_on_board.py','package_example.py','gallery-source.json'):
        files[name] = example/name
    for name in ('demo-image.jpg','imagenet-classes.txt','reference-logits.npy'):
        files['model/'+name] = example/'model'/name
    for p in (example/'model/aot').rglob('*'):
        if p.is_file() and 'ir' not in p.relative_to(example/'model/aot').parts:
            files['model/aot/'+p.relative_to(example/'model/aot').as_posix()] = p
    if not (example/'model/aot/pynq/model.so').is_file():
        raise RuntimeError('Compile the ARM package before board staging')
    files['execute_notebook.py'] = Path(__file__).with_name('execute_notebook.py')
    manifest = {'format':'npu-resnet18-aot/1','release_tag':'local-issue119-a-20261018',
                'source_commit':'beaf1f885360936e7dab97da6345190993c13e79',
                'working_tree':'uncommitted issue119-a refactor; per-file hashes identify the payload',
                'files':{name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sorted(files.items())}}
    dest = ROOT/'build/resnet18-board.tar.gz'
    receipt = ROOT/'build/board-package.manifest.json'
    receipt.write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(dest,'w:gz') as archive:
        for name,p in sorted(files.items()): archive.add(p,arcname=name,recursive=False)
        archive.add(receipt,arcname='package.manifest.json')
    print(dest, len(files), 'files; no JIT/export/checkpoint')


if __name__ == '__main__': main()
