"""Validate the arithmetic oracle against nearest representable-value search."""
import bisect
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.reference import multiply
# Positive finite values as exact integer multiples of minimum subnormal.
units=[(i&63) if (i>>6)==0 else (64+(i&63)) << ((i>>6)-1) for i in range(1984)]
# A virtual 65536 endpoint implements the overflow midpoint under RNE.
grid=[x << 20 for x in units]+[65536 << 40]
n=0
for a in range(1984):
    for b in range(a,1984):
        exact=units[a]*units[b]
        hi=bisect.bisect_left(grid,exact)
        if hi>=len(grid): expected=1984
        elif grid[hi]==exact: expected=hi
        elif hi==0: expected=0
        else:
            lo=hi-1;dl=exact-grid[lo];dh=grid[hi]-exact
            expected=lo if dl<dh or (dl==dh and not(lo&1)) else hi
        actual=multiply(a,b)
        if actual!=expected: raise AssertionError((a,b,actual,expected))
        n+=1
# Special classifications and sign symmetry for all encodings.
for a in range(4096):
    for b in [0,1,63,64,960,1983,1984,1985,2016,2048,4032,4095]:
        y=multiply(a,b)
        assert y==multiply(b,a)
        if y!=2016: assert multiply(a^2048,b)==(y^2048)
        n+=1
print(f'PASS independent nearest-grid oracle checks={n}')
