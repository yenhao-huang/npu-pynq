"""Re-elaborate historical SAT sources and audit their full packed interfaces.

This does not rerun SAT or retroactively alter any historical proof verdict.
Original files must still match the hashes in the recorded proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid
from ic_core.tools.exploration_common import fingerprint
from ic_core.tools.debug.formal import total_binary_module
from ic_core.tools.synth.acceptance import studies

here = Path(__file__).resolve().parent
os.chdir(here.parents[2])
parser = argparse.ArgumentParser()
parser.add_argument('--container', required=True)
args = parser.parse_args()
root = Path('docs/goals/0928-tool-exploration/evidence')
proofs = {}
for path in sorted(root.glob('*.json')):
    for study in studies(json.loads(path.read_text())):
        for case in study['data']['cases']:
            proof = case.get('formal', {})
            if proof.get('ok'):
                proofs[proof['run_id']] = proof

stage = here / 'output' / ('interface-audit-' + uuid.uuid4().hex[:12])
stage.mkdir(parents=True)
script, rows, excluded = [], [], []
for run_id, proof in sorted(proofs.items()):
    matches = list(Path('.ic/runs').glob('*/*-' + run_id + '-debug'))
    if len(matches) != 1:
        raise RuntimeError('Missing unique proof run: ' + run_id)
    meta = json.loads((matches[0] / 'meta.json').read_text())
    if meta['op'] != 'yosys_equivalence':
        excluded.append(dict(run_id=run_id, operation=meta['op']))
        continue
    params = meta['inputs']
    for role in ('reference', 'candidate'):
        files = [Path(p) for p in params[role + '_files']]
        source_hash = fingerprint(files, params['top'])
        if source_hash != proof['data'][role + '_sha256']:
            raise RuntimeError('Historical source changed: ' + run_id + '/' + role)
        name = run_id + '-' + role
        (stage / (name + '.sv')).write_text('\n'.join(p.read_text() for p in files))
        script.extend(['design -reset', 'read_verilog -sv ' + name + '.sv',
                       'hierarchy -check -top ' + params['top'],
                       'proc -noopt', 'flatten', 'check -assert',
                       'write_json ' + name + '.json'])
        rows.append(dict(run_id=run_id, role=role, source_sha256=source_hash,
                         top=params['top'], input_width=params['input_width'],
                         output_width=params['output_width'], name=name))
(stage / 'audit.ys').write_text('\n'.join(script) + '\n')
remote = '/tmp/ic-' + stage.name
commands = [['docker', 'exec', args.container, 'mkdir', remote],
            ['docker', 'cp', str(stage) + '/.', args.container + ':' + remote],
            ['docker', 'exec', '-w', remote, args.container, 'timeout', '180',
             'yosys', '-s', 'audit.ys'],
            ['docker', 'cp', args.container + ':' + remote + '/.', str(stage)]]
for i, command in enumerate(commands):
    with (stage / f'command-{i}.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                       timeout=210, check=True)
for row in rows:
    netlist = stage / (row.pop('name') + '.json')
    data = json.loads(netlist.read_text())
    ports = data['modules'][row['top']]['ports']
    row['ports'] = {name: dict(direction=p['direction'], width=len(p['bits']))
                    for name, p in ports.items()}
    row['interface_complete'] = row['ports'] == {
        'x': dict(direction='input', width=row['input_width']),
        'y': dict(direction='output', width=row['output_width'])}
    row['abstraction_defined'] = all(total_binary_module(module)
                                      for module in data['modules'].values())
    row['netlist_sha256'] = hashlib.sha256(netlist.read_bytes()).hexdigest()
    row['engine'] = data['creator']
result = dict(scope='Interface and total-binary source-netlist re-elaboration of source-bound successful SAT study proofs; no new cross-design SAT or physical verdict.',
              commands=commands, script_sha256=hashlib.sha256((stage / 'audit.ys').read_bytes()).hexdigest(),
              excluded_non_sat=excluded, rows=rows,
              passed=bool(rows) and all(r['interface_complete'] and r['abstraction_defined'] for r in rows))
(here / 'output' / 'historical-interface-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(len(rows), 'source interfaces', 'passed:', result['passed'], flush=True)
assert result['passed']
