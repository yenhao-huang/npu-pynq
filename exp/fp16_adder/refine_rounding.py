from run_eval import evaluate,ROOT
for name,mode,lzd in [('026_round_prefix','round_prefix','case'),('027_round_sentinel','round_sentinel','case'),('028_round_priority','round_sentinel','priority')]:
    if not (ROOT/'runs'/name).exists(): evaluate(name,mode,'operator',lzd,1000,constrained=True)
