"""Reject derived score overflow and recheck selection with actual study data."""
import json
import os
from pathlib import Path
from ic_core import dispatch
from ic_core.errors import InvalidInput

here=Path(__file__).resolve().parent
root=here.parents[2];os.chdir(root)
evidence=root/'docs/goals/0928-tool-exploration/evidence/wide-workflow.json'
rows={r['tool_id']:r for r in json.loads(evidence.read_text())['tools']}
reward=rows['08']['positive']['data']['reward'];cost=rows['10']['elapsed_s']
candidate=dict(name='repeat-csd-evaluation',mean_reward=reward,visits=1,estimated_cost_s=cost)
positive=dispatch('ppa_select',dict(candidates=[candidate],budget_s=cost*1.1))
assert positive['ok'] and positive['data']['selected']==candidate['name']
controls=[]
for label,value,expense,visits,exploration in [
    ('division_overflow',1e308,1e-308,1,1),
    ('addition_overflow',1e308,1,1,1e308),
    ('integer_conversion_overflow',1,1,10**1000,1),
]:
    payload=dict(candidates=[dict(name=label,mean_reward=value,visits=visits,estimated_cost_s=expense)],budget_s=1,exploration=exploration)
    try:dispatch('ppa_select',payload)
    except InvalidInput as error:
        assert 'finite numeric range' in str(error)
        controls.append(dict(label=label,payload=payload,rejected=True,reason=str(error)))
    else:raise AssertionError('Overflow accepted: '+label)
out=here/'output';out.mkdir(exist_ok=True)
(out/'finite-controls.json').write_text(json.dumps(dict(
    study_evidence=str(evidence.relative_to(root)),source_sha256=rows['09']['source_sha256'],
    reward_run=rows['08']['positive']['run_id'],measurement_pipeline=rows['10']['positive']['run_id'],
    positive=positive,negative_controls=controls,
    scope='Reuses the committed larger CSD study reward and measured cost; does not perform or claim a new physical measurement.'
),indent=2)+'\n')
print('Actual study selection passes; all three finite-input overflow controls rejected.')
