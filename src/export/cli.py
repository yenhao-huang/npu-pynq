"""Export a model into a package from its config.

    python -m src.export.cli build examples/smollm-safetensors/export.yml --out dist/smollm/package
    python -m src.export.cli build examples/resnet18/export.yml --out dist/resnet18/package
"""

from __future__ import annotations

import argparse
import json

from src.export import pipeline


def cmd_build(args) -> int:
    config = pipeline.read_config(args.config)
    timings: dict[str, float] = {}
    out = pipeline.export_package(config, args.out, source=args.source, timings=timings)
    print(json.dumps({'package': str(out), 'seconds': {k: round(v, 1) for k, v in timings.items()}}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='src.export.cli')
    sub = parser.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='Export a model from its config (examples/<demo>/export.yml)')
    b.add_argument('config')
    b.add_argument('--out', required=True)
    b.add_argument('--source', help="Override the config's model source")
    b.set_defaults(fn=cmd_build)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    raise SystemExit(main())
