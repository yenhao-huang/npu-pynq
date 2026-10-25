"""Download and verify the pinned SmolLM2 safetensors source."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

EXAMPLE = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=EXAMPLE / 'model' / 'checkpoint')
    args = parser.parse_args()
    spec = json.loads((EXAMPLE / 'model-source.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for name, info in spec['files'].items():
        dest = args.out / name
        if not dest.exists() or dest.stat().st_size != info['bytes'] or digest(dest) != info['sha256']:
            temporary = dest.with_suffix(dest.suffix + '.part')
            with urllib.request.urlopen(info['url'], timeout=300) as response, temporary.open('wb') as output:
                while block := response.read(1 << 20):
                    output.write(block)
            if temporary.stat().st_size != info['bytes'] or digest(temporary) != info['sha256']:
                raise ValueError('Source identity mismatch: ' + name)
            temporary.replace(dest)
        print('Verified', name, info['sha256'], flush=True)


if __name__ == '__main__':
    main()
