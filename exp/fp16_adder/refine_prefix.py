from run_eval import evaluate,ROOT
for name,mode,lzd in [('023_prefix','prefix','case'),('024_prefix_sentinel','prefix_sentinel','case'),('025_prefix_priority','prefix','priority')]:
    if not (ROOT/'runs'/name).exists(): evaluate(name,mode,'operator',lzd,1000,constrained=True)
