from run_eval import evaluate,ROOT
configs=[
('020_packed_csa','packed','csa','case',700),
('021_packed_split','packed','split','case',700),
('022_packed_priority','packed','operator','priority',700),
('023_prenorm_sized','prenorm','operator','priority',700),
('024_dual_sized','dual','operator','case',700),
('025_packed_rows','packed','rows','case',700),
]
for name,mode,mul,lzd,delay in configs:
    if (ROOT/'runs'/name/'metrics.json').exists():continue
    r=evaluate(name,mode,mul,lzd,delay,False,False,True)
    if r['status']!='pass':break
