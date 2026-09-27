from run_eval import evaluate,ROOT
for name,mode,lzd in [('020_sentinel','sentinel','case'),('021_sentinel_priority','sentinel','priority'),('022_parallel_exponent','prexp','case')]:
    if not (ROOT/'runs'/name).exists(): evaluate(name,mode,'operator',lzd,1000,constrained=True)
