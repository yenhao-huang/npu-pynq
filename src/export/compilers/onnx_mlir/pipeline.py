"""Inspect a source or import it to MLIR; does not create a runnable package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .support import inspect_source
from .toolchain import Compiler, OnnxMlirError


def prepare_import(source: str | Path, output: str | Path, *,
                   compiler: Compiler | None = None, inventory_only: bool = False) -> dict:
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Import output must be a new or empty directory')
    inventory = inspect_source(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    report = {'format': 'onnx-mlir-import/1', 'status': 'inventory-only',
              'source_sha256': inventory['source']['sha256'], 'compile_ready': False,
              'partition': 'not-run', 'codegen': 'not-run', 'board': 'not-run'}
    report_path = output / 'import.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if inventory_only:
        return report
    try:
        if inventory['external_data']:
            raise OnnxMlirError('External tensor data requires a verified source bundle before import')
        report.update((compiler or Compiler()).import_model(source, output))
        report['status'] = 'imported'
    except (OnnxMlirError, ValueError) as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--compiler', help='Explicit native binary; provenance remains unverified')
    args = parser.parse_args(argv)
    try:
        report = prepare_import(args.source, args.out, compiler=Compiler(binary=args.compiler),
                                inventory_only=args.inventory_only)
    except (OnnxMlirError, ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
