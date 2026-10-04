"""Second exploration round based on the measured normalization/rounding tradeoff."""
from run_eval import evaluate
CASES=[
('015_parallel_diff','parallel','case',1000,True),
('016_aligned_unconstrained','aligned','case',1000,False),
('017_shared_unconstrained','shared','priority',1000,False),
('018_parallel_unconstrained','parallel','case',1000,False),
('019_packed_priority','packed','priority',900,True),
]
if __name__=='__main__':
    for name,mode,lzd,delay,constrained in CASES:
        from run_eval import ROOT
        if (ROOT/'runs'/name).exists(): continue
        evaluate(name,mode,'operator',lzd,delay,constrained=constrained)
