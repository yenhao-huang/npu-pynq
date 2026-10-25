"""Execute the canonical notebook in a fresh board kernel; preserve failures."""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', type=Path, default=Path('smollm_safetensors.ipynb'))
    parser.add_argument('--out', type=Path, default=Path('smollm_safetensors.executed.ipynb'))
    args = parser.parse_args()
    nb = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name='python3',
                            resources={'metadata': {'path': str(args.notebook.resolve().parent)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, args.out)
    print('PASS:', args.out)


if __name__ == '__main__':
    main()
