"""Recorded hypothesis -> evaluation -> decision campaign; resume completed runs."""
import json
from pathlib import Path
from run_eval import evaluate
ROOT=Path(__file__).resolve().parent
CASES=[
('003_aligned_case','aligned','case',1200,'Parallel priority decode may shorten normalization.'),
('004_shared_priority','shared','priority',1200,'Share add/sub hardware to reduce area.'),
('005_shared_case','shared','case',1200,'Combine shared arithmetic with parallel leading-bit decode.'),
('006_staged_priority','staged','priority',1200,'Integrate sticky accumulation into the alignment barrel shifter.'),
('007_staged_case','staged','case',1200,'Combine staged sticky alignment with parallel decode.'),
('008_shared_900','shared','case',900,'Tighter mapping target explores the delay tradeoff.'),
('009_staged_900','staged','case',900,'Tighter mapping target on integrated alignment.'),
('010_shared_1600','shared','case',1600,'Relaxed mapping may reduce arithmetic area.'),
('011_staged_1600','staged','case',1600,'Relaxed mapping may reduce integrated alignment area.'),
('012_aligned_900','aligned','case',900,'Separate arithmetic paths may help delay at a tight target.'),
]
if __name__=='__main__':
    for name,mode,lzd,delay,hypothesis in CASES:
        path=ROOT/'runs'/name/'metrics.json'
        if path.exists():
            print('Existing record: '+name,flush=True);continue
        result=evaluate(name,mode,'operator',lzd,delay,constrained=True)
        result['hypothesis']=hypothesis
        result['decision']='Retain for joint area-delay and Pareto comparison.' if result['status']=='pass' else 'Exclude from ranking; inspect failure before reuse.'
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
