"""Seal the release's SmolLM package and require the board to reproduce it.

    reference  run the package on the host simulator (this release's own
               libnpu_rt SIM backend) and record the 32 greedy tokens
    seal       check the model's identity and record every file's SHA-256
    verify     check a deployed package against that record
    accept     generate on the physical NPU; the tokens must equal the
               reference, and the NPU must have done the contractions

The reference is produced by the same compiled package on the export host;
the board passes only when the FPGA, libnpu_rt's PYNQ backend and the ARM
CPU code reproduce it token for token.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
PROMPT = 'Write a paragraph about a neural processing unit (NPU).'
TOKENS = 32
SOURCE = REPO / 'examples/smollm-safetensors/model-source.json'
MODEL = 'smollm2-135m-instruct'


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
            if p.is_file() and p != package / 'release.json' and 'ir' not in p.relative_to(package).parts}


def identity(package):
    manifest = json.loads((package / 'manifest.json').read_text(encoding='utf-8'))
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    if (manifest.get('format') != 'pynqo-package/2' or manifest.get('model') != MODEL
            or manifest['llm']['cache']['layers'] != 30
            or not any(d['device'] == 'npu' for d in manifest['delegates'])):
        raise ValueError('Unexpected model identity')
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        if digest(package / name) != source['files'][name]['sha256']:
            raise ValueError('Tokenizer differs: ' + name)
    return source, manifest


def check_output(ids, text):
    if len(ids) <= 1 or not text.strip():
        raise ValueError('The example did not produce meaningful output.')


def reference(package):
    from src.runtime.tasks.llm import ExportedLLM
    identity(package)
    llm = ExportedLLM(package)
    ids, text, stats = llm.generate(PROMPT, max_new_tokens=TOKENS, chat=True)
    check_output(ids, text)
    write(package / 'reference.json', {'prompt': PROMPT, 'chat': True, 'max_new_tokens': TOKENS,
                                       'ids': ids, 'text': text, 'backend': 'sim', 'stats': stats})
    print(text)


def seal(package, commit, tag):
    identity(package)
    if not (package / 'reference.json').is_file():
        raise ValueError('Run reference before sealing')
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('An exact source commit is required')
    write(package / 'release.json', {'source_commit': commit, 'release_tag': tag, 'files': tree(package)})


def verify(package, commit, tag):
    record = json.loads((package / 'release.json').read_text(encoding='utf-8'))
    if record['source_commit'] != commit or record['release_tag'] != tag:
        raise ValueError('Package comes from a different release commit')
    if tree(package) != record['files']:
        raise ValueError('Package tree differs from the sealed build')
    identity(package)
    return record


def accept(args):
    from pynq import Overlay
    from src.runtime.tasks.llm import ExportedLLM
    package = args.package.resolve()
    release = verify(package, args.commit, args.tag)
    expected = json.loads((package / 'reference.json').read_text(encoding='utf-8'))
    overlay_path = args.overlay.resolve()
    if not overlay_path.with_suffix('.hwh').is_file():
        raise ValueError('The matching HWH is required')
    evidence = {'success': False, 'prompt': PROMPT, 'chat': True, 'max_new_tokens': TOKENS,
                'source_commit': args.commit, 'release_tag': args.tag,
                'deployment_id': args.deployment_id, 'reference': 'host simulator of this package',
                'package_files': release['files'],
                'overlay_sha256': digest(overlay_path),
                'hwh_sha256': digest(overlay_path.with_suffix('.hwh'))}
    try:
        llm = ExportedLLM(package, Overlay(str(overlay_path)))
        npu = llm.package.backends['npu']
        ids, text, stats = llm.generate(PROMPT, max_new_tokens=TOKENS, chat=True,
                                        on_token=lambda _, piece: print(piece, end='', flush=True))
        print(flush=True)
        runtime = stats['npu']
        evidence.update(ids=ids, text=text, stats=stats, runtime=runtime,
                        backend=1 if npu.platform is not None else 0, mmio=npu.mmio)
        check_output(ids, text)
        if ids != expected['ids'] or text != expected['text']:
            raise ValueError('SmolLM token IDs or text differ from the host reference')
        if (npu.platform is None or not npu.mmio or runtime['calls'] <= 0
                or runtime['calls'] != runtime['delegate_calls'] or runtime['jobs'] <= 0
                or runtime['cycles'] <= 0 or npu.stats()['error'] != 0):
            raise ValueError('Physical NPU execution was not proven')
        evidence['success'] = True
    except Exception as error:
        evidence['error'] = str(error)
        raise
    finally:
        write(args.evidence, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('reference', 'seal', 'verify', 'accept'))
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--commit')
    parser.add_argument('--tag')
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--deployment-id')
    args = parser.parse_args()
    if args.command == 'reference':
        reference(args.package.resolve())
        return
    if not (args.commit and args.tag):
        parser.error(f'{args.command} requires --commit and --tag')
    if args.command == 'accept':
        if not all((args.overlay, args.evidence, args.deployment_id)):
            parser.error('accept requires --overlay, --evidence and --deployment-id')
        accept(args)
    else:
        globals()[args.command](args.package.resolve(), args.commit, args.tag)


if __name__ == '__main__':
    main()
