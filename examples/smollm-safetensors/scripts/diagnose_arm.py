"""Run the same ARM library with the source ISA simulator, without the FPGA."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--tokenwise', action='store_true')
    parser.add_argument('--package', default='package')
    parser.add_argument('--prompt', default='write a paragraph about npu')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    package = root / args.package
    reference = root / ('arm-sim-' + args.package)
    reference.mkdir(exist_ok=True)
    for path in package.iterdir():
        if path.name != 'manifest.json' and not (reference / path.name).exists():
            (reference / path.name).symlink_to(path, target_is_directory=path.is_dir())
    manifest = json.loads((package / 'manifest.json').read_text())
    manifest['libraries'] = {'pynq': manifest['libraries']['pynq']}
    (reference / 'manifest.json').write_text(json.dumps(manifest))
    from src.runtime.core.engine import LLMEngine
    engine = LLMEngine(reference, backend='sim', stream_weights=True)
    ids, text, stats = engine.generate(args.prompt, max_new_tokens=32, chat=True,
                                      use_chunked_prefill=not args.tokenwise)
    record = {'prompt': args.prompt, 'output_ids': ids, 'output': text,
              'stats': stats.as_dict(), 'runtime': engine.package.stats()}
    name = 'arm-sim' + ('-tokenwise' if args.tokenwise else '')
    name += ('-' + args.package if args.package != 'package' else '') + '-result.json'
    (root / name).write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
