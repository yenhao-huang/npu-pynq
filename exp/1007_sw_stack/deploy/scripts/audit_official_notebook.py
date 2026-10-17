"""Audit physical notebook outputs against the unchanged full ARM ONNX oracle."""
import argparse
import json
from pathlib import Path
import nbformat
import numpy as np
from acceptance import save, sha256, MAX_NEW
from official_acceptance import SOURCE_SHA256, compare


def audit(deploy, run_id):
    run = deploy / 'runs' / run_id
    named = deploy / 'smollm_board_onnx_quant.ipynb'
    executed = run / 'smollm_board_onnx_quant.executed.ipynb'
    nb = nbformat.read(executed, as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == 'code']
    errors = [dict(o) for c in code for o in c.outputs if o.output_type == 'error']
    pngs = sum('image/png' in o.get('data', {}) for c in code for o in c.outputs)
    read = lambda name: json.loads((run / (name + '.json')).read_text())
    execution = read('notebook-audit')
    verdict = read('verdict')
    result = read('generation')
    interactive = read('interactive')
    package = read('package')
    load = read('load')
    arithmetic = read('gemm')
    manifest = json.loads((deploy / 'packages/onnx-graph/graph.json').read_text())
    arm = deploy / 'data/official-arm-oracle'
    arm_info = json.loads((arm / 'summary.json').read_text())
    arm_comparison = compare(run / 'generation.npz', arm / 'reference.npz')
    cpu_comparison = compare(run / 'generation.npz', run / 'cpu-reference.npz')
    repeat_comparison = compare(run / 'generation.npz', run / 'repeat.npz')
    with np.load(run / 'generation.npz') as trace:
        durable_ids_equal = np.array_equal(trace['ids'], result['ids'])
        finite = bool(np.isfinite(trace['logits']).all())
    gates = {
        'all_eight_cells_executed': len(code) == 8 and
            [c.execution_count for c in code] == list(range(1, 9)),
        'all_cells_have_outputs': all(c.outputs for c in code),
        'no_notebook_errors': not errors,
        'executor_success': execution['success'] and execution['executed_code_cells'] == 8,
        'named_file_is_executed_file': sha256(named) == sha256(executed),
        'saved_figures': pngs >= 4,
        'official_source_sha256': sha256(deploy / 'packages/onnx/model_int8.onnx') ==
            manifest['source_sha256'] == package['source_sha256'] == SOURCE_SHA256,
        'official_source_bytes': package['official_bytes'] == 137147867,
        'lossless_tensor_sha256': sha256(deploy / 'packages/onnx-graph/tensors.bin') ==
            manifest['tensors_sha256'],
        'full_model': package['layers'] == 30 and package['integer_projections'] == 210,
        'native_arm_runtime': load['onnxruntime_version'] == arm_info['runtime'] == '1.20.1',
        'same_arm_runtime_binary': sha256(deploy / 'packages/onnx-runtime/libonnxruntime.so.1.20.1') ==
            arm_info['runtime_sha256'],
        'unchanged_arm_oracle': arm_info['unchanged_onnx'] and
            arm_info['source_sha256'] == SOURCE_SHA256 and arm_info['steps'] == MAX_NEW and
            arm_info['prompt_tokens'] == len(result['prompt_ids']) and
            arm_info['prompt_ids'] == result['prompt_ids'] and
            sha256(arm / 'reference.npz') == arm_info['reference_sha256'],
        'exact_physical_arithmetic': len(arithmetic) == 3 and all(c['exact'] for c in arithmetic),
        'all_integer_nodes_physical': result['npu']['calls'] == 210 * MAX_NEW,
        'physical_jobs_and_cycles': result['npu']['jobs'] > 0 and result['npu']['cycles'] > 0,
        'notebook_verdict': verdict['pass'] and all(verdict.values()),
        'durable_generation': bool(durable_ids_equal) and finite and len(result['ids']) == MAX_NEW,
        'unchanged_arm_tokens': arm_comparison['token_ids_equal'],
        'unchanged_arm_logits': arm_comparison['logits_allclose'],
        'cpu_reference_tokens': cpu_comparison['token_ids_equal'],
        'cpu_reference_logits': cpu_comparison['logits_allclose'],
        'repeat_tokens': repeat_comparison['token_ids_equal'],
        'repeat_logits': repeat_comparison['logits_allclose'],
        'editable_prompt_generation': len(interactive['ids']) > 1 and bool(interactive['text'].strip()),
    }
    evidence = {
        'run_id': run_id, 'pass': all(gates.values()), 'gates': gates,
        'execution_counts': [c.execution_count for c in code], 'errors': errors,
        'png_outputs': pngs, 'arm_comparison': arm_comparison,
        'cpu_comparison': cpu_comparison, 'repeat_comparison': repeat_comparison,
        'notebook_sha256': sha256(named), 'arm_reference_sha256': sha256(arm / 'reference.npz'),
        'timing': {k: result[k] for k in ('ttft_seconds', 'tpot_seconds', 'decode_tokens_per_second')},
        'physical_npu': result['npu'], 'interactive_tokens': len(interactive['ids']),
    }
    save(run / 'final-audit.json', evidence)
    print(json.dumps(evidence, indent=2))
    assert evidence['pass'], 'Final acceptance audit failed; inspect final-audit.json.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    audit(Path(__file__).resolve().parents[1], args.run_id)
