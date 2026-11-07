"""Verify a sealed release bundle (src/deploy/bundle.py) without host compiler dependencies."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'package.manifest.json').read_text())
    if manifest.get('format') != 'npu-bundle/1':
        raise ValueError('Expected a sealed npu-bundle/1 release')
    for name, expected in manifest['files'].items():
        candidate = root/name
        path = candidate.resolve()
        if candidate.is_symlink() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or unsafe delivery file: '+name)
        if sha256_file(path) != expected:
            raise ValueError('Delivery hash mismatch: '+name)
    return manifest
