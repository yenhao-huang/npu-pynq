"""Cross-check exact arithmetic with Python binary64 plus binary16 packing."""
import sys
from pathlib import Path
import random
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.reference import add,native
r=random.Random(160527)
anchors=[0,0x8000,1,0x3ff,0x400,0x3c00,0x7bff,0x7c00,0xfc00,0x7e00]
count=0
for a in range(65536):
    for b in anchors+[a,a^0x8000,r.randrange(65536)]:
        assert add(a,b)==native(a,b),(hex(a),hex(b),hex(add(a,b)),hex(native(a,b)))
        count+=1
print(f'PASS independent reference checks={count}')
