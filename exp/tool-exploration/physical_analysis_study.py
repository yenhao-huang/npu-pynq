"""Reanalyze recorded physical evidence without invoking implementation tools."""
import json
from pathlib import Path
from ic_core import dispatch
root=Path(__file__).resolve().parents[2]
study=json.loads((root/'docs/goals/0928-tool-exploration/evidence/priority-physical-study.json').read_text())
for case in study['data']['cases']:
    pairs=[dict(baseline=pair[0],candidate=pair[1]) for pair in case['records']]
    for tool_id,op,payload in [('45','resource_tradeoff',pairs[0]),('46','paired_repeat_summary',dict(pairs=pairs,objective=case['objective']))]:
        result=dispatch(op,payload,cwd=root)
        output=root/f'exp/tool-exploration/exp-tool-{tool_id}-aspen/output'
        output.mkdir(exist_ok=True)
        (output/(case['name']+'-'+result['run_id']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(op,case['name'],result['data']['area_gate'],result['data']['throughput_gate'])
