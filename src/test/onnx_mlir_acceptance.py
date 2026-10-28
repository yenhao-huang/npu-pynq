"""Independent ONNX reference fixtures and compiled SmolLM board acceptance.

Generate fixtures on a host with ONNX Runtime 1.20.1. Check them on either the
simulated NPU or PYNQ-Z1 without installing ONNX Runtime on the board.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from src.runtime.core.engine import LLMEngine
from src.runtime.core.tokenizer import Tokenizer


TOLERANCES = {'logits': {'rtol': 1e-3, 'atol': 1e-2},
              'cache': {'rtol': 1e-3, 'atol': 2e-3}}
PROMPT = 'Write a paragraph about a neural processing unit (NPU).'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def fixtures(source, package, output, decode_steps=3):
    import onnxruntime as ort
    if ort.__version__ != '1.20.1':
        raise ValueError('Reference fixtures require ONNX Runtime 1.20.1')
    if decode_steps < 2:
        raise ValueError('Acceptance requires multiple decode steps')
    source, package, output = Path(source), Path(package), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Reference output must be a new or empty directory')
    manifest = json.loads((package / 'manifest.json').read_text())
    source_hash = digest(source)
    if source_hash != manifest['meta']['source_sha256'] or manifest['meta']['layers'] != 30:
        raise ValueError('Source identity or full-model layer count differs')
    output.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(source), options, providers=['CPUExecutionProvider'])
    names = [value.name for value in session.get_outputs()]
    tokenizer = Tokenizer(package)
    prompt = tokenizer.encode(tokenizer.chat([{'role': 'user', 'content': PROMPT}]))
    if len(prompt) != 42:
        raise ValueError('Pinned tokenizer/chat profile must contain 42 tokens')
    cases = []
    for length in (36, 40, 42):
        past = {f'past_key_values.{i}.{kind}': np.empty((1, 3, 0, 64), 'f4')
                for i in range(30) for kind in ('key', 'value')}
        ids = prompt[:length]
        for step in range(decode_steps + 1):
            position = next(iter(past.values())).shape[2]
            inputs = {'input_ids': np.array([ids], 'i8'),
                'attention_mask': np.ones((1, position + len(ids)), 'i8'),
                'position_ids': np.arange(position, position + len(ids), dtype='i8')[None], **past}
            start = time.monotonic()
            result = dict(zip(names, session.run(None, inputs)))
            seconds = time.monotonic() - start
            past = {name.replace('present.', 'past_key_values.'): value
                    for name, value in result.items() if name != 'logits'}
            logits = result['logits'][:, -1]
            caches = [np.concatenate([past[f'past_key_values.{i}.{kind}'] for i in range(30)])
                      for kind in ('key', 'value')]
            path = output / f'prefill{length}-step{step}.npz'
            np.savez(path, input_ids=np.array(ids, 'i4'), logits=logits, key=caches[0], value=caches[1])
            cases.append({'length': length, 'step': step, 'position': position,
                          'file': path.name, 'sha256': digest(path), 'reference_seconds': seconds,
                          'next_token': int(logits.argmax())})
            ids = [int(logits.argmax())]
    record = {'format': 'onnx-mlir-reference/1', 'source_sha256': source_hash,
              'reference': 'onnxruntime-1.20.1-full-source', 'tolerances': TOLERANCES, 'cases': cases}
    write(output / 'reference.json', record)
    print('ONNX_REFERENCE_FIXTURES_PASS', flush=True)
    return record


def check(package, fixture_dir, evidence, backend='sim', overlay=None):
    package, fixture_dir, evidence = Path(package), Path(fixture_dir), Path(evidence)
    reference = json.loads((fixture_dir / 'reference.json').read_text())
    manifest = json.loads((package / 'manifest.json').read_text())
    if reference['format'] != 'onnx-mlir-reference/1' or reference['tolerances'] != TOLERANCES:
        raise ValueError('Reference format or numerical tolerances differ')
    if manifest['meta'].get('source_sha256') != reference['source_sha256']:
        raise ValueError('Compiled package and reference source identities differ')
    if manifest['meta'].get('native_float_runtime') or manifest['meta'].get('runtime_hooks'):
        raise ValueError('Acceptance requires an ORT-free compiled package')
    if backend == 'pynq':
        from pynq import Overlay
        if not overlay:
            raise ValueError('Physical acceptance requires an overlay path')
        overlay = Overlay(str(overlay))
    engine = LLMEngine(package, backend, overlay=overlay, stream_weights=True)
    record = {'status': 'running', 'backend': backend, 'source_sha256': reference['source_sha256'],
              'manifest_sha256': digest(package / 'manifest.json'), 'tolerances': TOLERANCES, 'cases': []}
    evidence.parent.mkdir(parents=True, exist_ok=True)
    try:
        for case in reference['cases']:
            path = fixture_dir / case['file']
            if path.parent.resolve() != fixture_dir.resolve() or digest(path) != case['sha256']:
                raise ValueError('Reference tensor fixture identity differs')
            if case['step'] == 0:
                engine.reset()
            with np.load(path) as fixture:
                ids = fixture['input_ids'].tolist()
                if engine.pos != case['position']:
                    raise ValueError('Reference and compiled cache positions differ')
                before = engine.package.stats()
                start = time.monotonic()
                actual = engine.prefill(ids) if case['step'] == 0 else engine.decode(ids[0])
                seconds = time.monotonic() - start
                stats = engine.package.stats()
                errors = {'logits': float(np.max(np.abs(actual - fixture['logits'])))}
                for kind, state in zip(('key', 'value'), engine.state):
                    errors[kind] = float(np.max(np.abs(state[:, :, :engine.pos] - fixture[kind])))
                observed = {**case, 'seconds': seconds, 'errors': errors,
                    'token': int(actual.argmax()), 'npu_delta': {key: stats[key] - before[key]
                        for key in ('calls', 'jobs', 'cycles', 'macs', 'polls')},
                    'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                record['cases'].append(observed)
                write(evidence, record)
                print(json.dumps(observed), flush=True)
                np.testing.assert_allclose(actual, fixture['logits'], **TOLERANCES['logits'])
                for kind, state in zip(('key', 'value'), engine.state):
                    np.testing.assert_allclose(state[:, :, :engine.pos], fixture[kind], **TOLERANCES['cache'])
                    np.testing.assert_array_equal(state[:, :, engine.pos:], 0)
                if actual.argmax() != case['next_token']:
                    raise AssertionError('Deterministic next token differs')
                if stats['error'] or observed['npu_delta']['jobs'] <= 0:
                    raise AssertionError('NPU did not complete the expected work')
                if backend == 'pynq' and observed['npu_delta']['cycles'] <= 0:
                    raise AssertionError('Physical NPU cycles were not observed')
        covered = {(case['length'], case['step']) for case in reference['cases']}
        if not {(length, step) for length in (36, 40, 42) for step in range(3)} <= covered:
            raise AssertionError('Required prefill and decode coverage is incomplete')
        record['status'] = 'passed'
    except Exception as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        write(evidence, record)
        for name in ('weights', 'io', 'stage', 'words'):
            buffer = getattr(engine.package, name, None)
            if buffer is not None and hasattr(buffer, 'buffer'):
                buffer.buffer.freebuffer()
    print('PHYSICAL_SMOLLM_ONNX_MLIR_PASS' if backend == 'pynq' else 'HOST_SMOLLM_ONNX_MLIR_PASS', flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('fixtures')
    generate.add_argument('--source', required=True)
    generate.add_argument('--package', required=True)
    generate.add_argument('--out', required=True)
    verify = commands.add_parser('check')
    verify.add_argument('--package', required=True)
    verify.add_argument('--fixtures', required=True)
    verify.add_argument('--evidence', required=True)
    verify.add_argument('--backend', choices=('sim', 'pynq'), default='sim')
    verify.add_argument('--overlay')
    args = parser.parse_args(argv)
    if args.command == 'fixtures':
        fixtures(args.source, args.package, args.out)
    else:
        check(args.package, args.fixtures, args.evidence, args.backend, args.overlay)


if __name__ == '__main__':
    main()
