"""Seal release model assets and require deterministic physical SmolLM output."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
PROMPT = 'Write a paragraph about a neural processing unit (NPU).'
# Reviewed from the unchanged full pinned ONNX with ORT 1.20.1 on 2026-10-25,
# before checking the candidate's 32-token notebook run. Exact input IDs and
# provenance are recorded in docs/plans/onnx-mlir-incoporate/acceptance.md.
# Never regenerate this baseline from a candidate under test.
EXPECTED_IDS = [49,25934,18784,4730,365,26282,69,25,314,253,1502,282,2583,817,
                338,3078,1096,281,253,7623,5710,28,1015,2701,4604,4784,355,
                21443,288,1477,253,3599]
EXPECTED_TEXT = ('A Neural Processing Unit (NPU) is a type of computer system that '
                 'processes information in a parallel manner, using multiple '
                 'processing units or cores to perform a wide')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def tree(package):
    return {p.relative_to(package).as_posix(): digest(p)
            for p in sorted(package.rglob('*'))
            if p.is_file() and p != package / 'release.json'}


def identity(package):
    manifest = json.loads((package / 'manifest.json').read_text(encoding='utf-8'))
    source = json.loads((REPO / 'src/export/graphs/smollm/onnx-source.json').read_text(encoding='utf-8'))
    meta = manifest['meta']
    if (not (meta['family'] == 'official-smollm-onnx' or
             (meta['family'] == 'llama' and meta.get('source_format') == 'onnx')) or meta['layers'] != 30
            or meta['source_sha256'] != source['files']['model_int8.onnx']['sha256']
            or 42 not in meta['prefill_lengths']):
        raise ValueError('Unexpected model identity or missing complete chat prefill')
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        if digest(package / name) != source['files'][name]['sha256']:
            raise ValueError('Tokenizer differs: ' + name)
    return source, manifest


def seal(package, commit, tag):
    identity(package)
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('An exact source commit is required')
    write(package / 'release.json', {'source_commit': commit, 'release_tag': tag,
                                    'files': tree(package)})


def verify(package, commit, tag):
    record = json.loads((package / 'release.json').read_text(encoding='utf-8'))
    if record['source_commit'] != commit or record['release_tag'] != tag:
        raise ValueError('Package comes from a different release commit')
    if tree(package) != record['files']:
        raise ValueError('Package tree differs from the sealed build')
    identity(package)
    return record


def check_output(ids, text):
    if len(ids) <= 1 or not text.strip():
        raise ValueError('The interactive example did not produce meaningful output.')
    if ids != EXPECTED_IDS or text != EXPECTED_TEXT:
        raise ValueError('SmolLM token IDs or text differ from the frozen reference')


def accept(args):
    from pynq import Overlay
    from src.runtime.core.engine import LLMEngine
    package = args.package.resolve()
    release = verify(package, args.commit, args.tag)
    overlay_path = args.overlay.resolve()
    if not overlay_path.with_suffix('.hwh').is_file():
        raise ValueError('The matching HWH is required')
    evidence = {'success': False, 'prompt': PROMPT, 'chat': True, 'max_new_tokens': 32,
                'source_commit': args.commit, 'release_tag': args.tag,
                'deployment_id': args.deployment_id,
                'reference_run': 'full-source-onnx-20261025',
                'package_files': release['files'],
                'overlay_sha256': digest(overlay_path),
                'hwh_sha256': digest(overlay_path.with_suffix('.hwh'))}
    try:
        overlay = Overlay(str(overlay_path))
        engine = LLMEngine(package, backend='pynq', overlay=overlay)
        before = engine.package.stats()
        ids, text, stats = engine.generate(PROMPT, max_new_tokens=32, chat=True,
            on_token=lambda _, piece: print(piece, end='', flush=True))
        print(flush=True)
        after = engine.package.stats()
        runtime = {k: after[k] - before[k] for k in ('calls', 'jobs', 'cycles', 'macs')}
        runtime.update(error=after['error'], error_program=after['error_program'])
        # The runtime exposes its selected backend and actual MMIO pointer.
        context = engine.package.lib.npu_rt_stats().contents
        evidence.update(ids=ids, text=text, stats=stats, runtime=runtime,
                        backend=context.backend, mmio=int(context.mmio or 0))
        check_output(ids, text)
        if (engine.package.backend != 'pynq' or context.backend != 1 or not context.mmio
                or runtime['calls'] != 32 * 120 or runtime['jobs'] <= 0
                or runtime['cycles'] <= 0 or runtime['error'] != 0):
            raise ValueError('Physical NPU execution was not proven')
        evidence['success'] = True
    except Exception as error:
        evidence['error'] = str(error)
        raise
    finally:
        write(args.evidence, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('seal', 'verify', 'accept'))
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--deployment-id')
    args = parser.parse_args()
    if args.command == 'accept':
        if not all((args.overlay, args.evidence, args.deployment_id)):
            parser.error('accept requires --overlay, --evidence and --deployment-id')
        accept(args)
    else:
        globals()[args.command](args.package.resolve(), args.commit, args.tag)


if __name__ == '__main__':
    main()
