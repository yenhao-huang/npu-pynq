"""A board bundle: what the PYNQ-Z1 runs for one npu-package/2 package.

    <out>/package/            the package with only the board's CPU library
    <out>/src/...             the runtime and the NPU driver sources (built on the board)
    <out>/npu_matrix.bit/.hwh the overlay
    <out>/<extra files>       notebooks, labels, images

A release bundle is the same tree sealed by ``seal``: package.manifest.json
names the release, the source and overlay commits and every file's SHA-256;
src/runtime/share/bundle.py verifies it on the board before anything runs.
``archive`` writes it as a deterministic zip.

On the board, from the bundle directory, the runtime runs with the PYNQ
environment as root (overlay loading and MMIO), for example:

    sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli llm package ...
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

REPO = Path(__file__).resolve().parents[2]
BOARD_CPU = 'arm-cortex-a9'
RUNTIME = ('src/runtime', 'src/vendors/__init__.py', 'src/vendors/npu_pynq/__init__.py',
           'src/vendors/npu_pynq/driver', 'src/vendors/npu_pynq/spec')
SKIP = shutil.ignore_patterns('__pycache__', '*.pyc')
RELEASE_FORMAT = 'npu-bundle/1'


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
    for name in ('npu_matrix.bit', 'npu_matrix.hwh', 'npu_matrix.manifest.json'):
        if (overlay / name).is_file():
            shutil.copy2(overlay / name, out / name)
    for rel, src in (extra or {}).items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / rel)
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def seal(out: Path, release_tag: str, source_commit: str, overlay_commit: str | None = None,
         reports: Path | None = None) -> dict:
    """Check the overlay against its build manifest and the package, then write package.manifest.json."""
    from src.vendors.npu_pynq.driver.overlay.verify import verify_artifacts

    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('an exact 40-character source commit is required')
    overlay = verify_artifacts(out)
    if overlay['source_commit'] != (overlay_commit or source_commit):
        raise ValueError('the overlay was built from a different commit than declared')
    limits = json.loads((out / 'package' / 'manifest.json').read_text()).get('npu_limits', {})
    if limits.get('rows') != overlay['array_size'] or limits.get('columns') != overlay['array_size']:
        raise ValueError('the package was compiled for a different array than the overlay')
    if reports:
        shutil.copytree(reports, out / 'reports', dirs_exist_ok=True)
    files = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.name != 'package.manifest.json'}
    manifest = {'format': RELEASE_FORMAT, 'release_tag': release_tag, 'source_commit': source_commit,
                'overlay_commit': overlay_commit or source_commit, 'files': files}
    (out / 'package.manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


def archive(out: Path, path: Path) -> Path:
    """The bundle as a zip with fixed timestamps and modes (same input, same bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                info = zipfile.ZipInfo(p.relative_to(out).as_posix(), (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                zf.writestr(info, p.read_bytes())
    return path
