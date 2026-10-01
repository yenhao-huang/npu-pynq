"""Independent exact binary16 addition in units of the minimum subnormal."""
import struct


def add(a, b):
    def decode(x):
        e=(x>>10)&31
        return ((x&1023)|(1024 if e else 0)) << max(0,e-1)
    ea=(a>>10)&31; eb=(b>>10)&31
    if (ea==31 and a&1023) or (eb==31 and b&1023): return 0x7e00
    if ea==31 and eb==31 and (a^b)&0x8000: return 0x7e00
    if ea==31: return a
    if eb==31: return b
    s=(-decode(a) if a&0x8000 else decode(a))+(-decode(b) if b&0x8000 else decode(b))
    if not s: return (a&b)&0x8000
    sign=0x8000 if s<0 else 0; n=abs(s)
    shift=max(0,n.bit_length()-11)
    q,r=divmod(n,1<<shift)
    if shift and (r>(1<<(shift-1)) or (r==(1<<(shift-1)) and q&1)): q+=1
    if q==2048: q>>=1; shift+=1
    if shift>=30: return sign|0x7c00
    return sign | (q if q<1024 else ((shift+1)<<10)|(q&1023))


def native(a,b):
    x=struct.unpack('<e',struct.pack('<H',a))[0]
    y=struct.unpack('<e',struct.pack('<H',b))[0]
    try: z=struct.unpack('<H',struct.pack('<e',x+y))[0]
    except OverflowError: z=0xfc00 if x+y<0 else 0x7c00
    return 0x7e00 if z&0x7c00==0x7c00 and z&1023 else z
