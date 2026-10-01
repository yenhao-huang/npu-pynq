from run_eval import evaluate,ROOT
configs=[
('027_parallel_priority','parallel','operator','priority',700),
('028_parallel_csa','parallel','csa','case',700),
('029_packed_priority_900','packed','operator','priority',900),
('030_packed_priority_1200','packed','operator','priority',1200),
('031_packed_priority_300','packed','operator','priority',300),
('032_baseline_sized','baseline','operator','priority',700),
]
for name,mode,mul,lzd,delay in configs:
    if (ROOT/'runs'/name/'metrics.json').exists():continue
    r=evaluate(name,mode,mul,lzd,delay,False,False,True)
    if r['status']!='pass':break
