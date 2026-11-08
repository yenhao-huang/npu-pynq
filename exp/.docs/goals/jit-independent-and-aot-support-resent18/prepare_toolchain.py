"""Extract official LLVM 22 Debian packages into an ignored local directory.

No apt sources, system packages or system paths are modified. Run on Debian 12
amd64; package hashes are verified against the official repository index and
recorded for reproduction. This script is an experiment environment helper.
"""
import gzip
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

BASE = 'https://apt.llvm.org/bookworm/'
NAMES = ('libllvm22', 'llvm-22', 'llvm-22-runtime', 'llvm-22-linker-tools',
         'llvm-22-dev', 'libclang-cpp22', 'libclang-common-22-dev',
         'libclang1-22', 'clang-22', 'libmlir-22', 'libmlir-22-dev',
         'mlir-22-tools', 'lld-22')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--lock',type=Path,help='Use previously recorded package versions and hashes')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.lock:
        records = json.loads(args.lock.read_text())
    else:
        index = gzip.decompress(urllib.request.urlopen(
            BASE+'dists/llvm-toolchain-bookworm-22/main/binary-amd64/Packages.gz').read()).decode()
        records = {}
        for block in index.split('\n\n'):
            fields = dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line and not line.startswith(' '))
            if fields.get('Package') in NAMES:
                records[fields['Package']] = fields
    if set(NAMES) - records.keys():
        raise RuntimeError('Missing packages: '+str(set(NAMES)-records.keys()))
    for name in NAMES:
        record = records[name]
        dest = root / Path(record['Filename']).name
        if not dest.exists():
            print('Downloading', name, record['Size'], flush=True)
            urllib.request.urlretrieve(BASE+record['Filename'], dest)
        if hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest() != record['SHA256']:
            raise RuntimeError('Package hash mismatch: '+name)
        subprocess.run(['dpkg-deb', '-x', str(dest), str(root/'root')], check=True)
    (root/'packages.json').write_text(json.dumps({name:{k:records[name][k] for k in
        ('Version','Filename','Size','SHA256')} for name in NAMES}, indent=2)+'\n')
    print('Extracted LLVM 22 to', root/'root/usr/lib/llvm-22/bin', flush=True)


if __name__ == '__main__':
    main()
