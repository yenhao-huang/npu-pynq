"""Signed matrix arithmetic oracle independent of PE forwarding or schedule."""
from .cycles import transaction_trace


def pack(values,width):
    return sum((value&((1<<width)-1))<<(i*width) for i,value in enumerate(values))


def matrix_result(data,width,size):
    mask=(1<<width)-1
    words=[(data>>(i*width))&mask for i in range(2*size*size)]
    signed=[v-(1<<width) if v&(1<<(width-1)) else v for v in words]
    a,b=signed[:size*size],signed[size*size:]
    result=[sum(a[i*size+k]*b[k*size+j] for k in range(size)) for i in range(size) for j in range(size)]
    return pack(result,2*width+(size-1).bit_length()),not any(a) or not any(b)


def matrix_trace(width,size,latency,random_cycles,seed):
    count=size*size;minimum=-(1<<(width-1));maximum=(1<<(width-1))-1
    identity=[int(i==j) for i in range(size) for j in range(size)]
    nonsymmetric=[(i+1)*(-1 if i%2 else 1) for i in range(count)]
    cases=[([0]*count,nonsymmetric),(nonsymmetric,identity),(identity,nonsymmetric),
           ([minimum]*count,[minimum]*count),([minimum]*count,[maximum]*count),
           ([maximum]*count,[maximum]*count),(nonsymmetric,list(reversed(nonsymmetric)))]
    # Every individual row/column route receives a nonzero one-hot contribution.
    for i in range(size):
        for k in range(size):
            for j in range(size):
                a=[0]*count;b=[0]*count;a[i*size+k]=-3;b[k*size+j]=5;cases.append((a,b))
    transactions=[pack(a+b,width) for a,b in cases]
    return transaction_trace(2*count*width,transactions,lambda data:matrix_result(data,width,size),latency,random_cycles,seed)
