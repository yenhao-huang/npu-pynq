"""Run one screening round; immutable run directories enable restart."""
import json
from pathlib import Path
from run_eval import evaluate,ROOT
configs=[
('005_baseline_case','baseline','operator','case',1200,False),
('006_compact_case','compact','operator','case',1200,False),
('007_compact_csa','compact','csa','case',1200,False),
('008_compact_rows','compact','rows','case',1200,False),
('009_compact_split','compact','split','case',1200,False),
('010_baseline_csa','baseline','csa','case',1200,False),
('011_baseline_rows','baseline','rows','case',1200,False),
('012_baseline_split','baseline','split','case',1200,False),
('013_compact_900','compact','operator','case',900,False),
('014_compact_1700','compact','operator','case',1700,False),
('015_compact_fast','compact','operator','case',1200,True),
]
for name,mode,mul,lzd,delay,fast in configs:
    if (ROOT/'runs'/name/'metrics.json').exists():
        print('Existing run retained:',name,flush=True);continue
    result=evaluate(name,mode,mul,lzd,delay,False,fast)
    if result['status']=='failed':
        print('Campaign paused at failed configuration; inspect evidence before resuming.',flush=True);break
