"""Queue-per-stage model with independent end-to-end ordering and cycle checks."""
from collections import deque
import random
from ...errors import InvalidInput


def stream_vectors(width,stages,architecture,cycles,seed):
    if not 8<=width<=64 or not 2<=stages<=32: raise InvalidInput('Stream width 8..64 and stage count 2..32 required')
    rng=random.Random(seed);queues=[deque() for _ in range(stages)];end_to_end=deque()
    slots=2 if architecture=='skid' else 1
    calibration_end=128+stages+2;stall_end=calibration_end+4*stages+4
    random_end=stall_end+stages+4+cycles
    pending=None;vectors=[];accepted=[];calibration_latencies=[]
    coverage=dict(push=0,pop=0,simultaneous=0,input_stall=0,output_stall=0,reset_nonempty=0)
    for cycle in range(random_end+4*stages+4):
        rst=int(cycle in (0,stall_end,stall_end+stages+2) or (stall_end+stages+3<cycle<random_end and cycle%997==0))
        ready_out=0 if calibration_end<=cycle<stall_end else (1 if cycle<stall_end+stages+4 or cycle>=random_end else int(rng.random()<.6))
        valid_in=int(cycle<random_end and (cycle<stall_end+stages+4 or rng.random()<.8))
        if pending is not None: valid_in=1
        value=pending if pending is not None else rng.getrandbits(width)
        ready=[0]*stages;valid=[int(bool(q) and not rst) for q in queues]
        for i in range(stages-1,-1,-1):
            downstream=ready_out if i==stages-1 else ready[i+1]
            ready[i]=int(not rst and (len(queues[i])<slots or (architecture=='elastic' and downstream)))
        output=queues[-1][0] if queues[-1] else None
        vectors.append((rst,valid_in,ready_out,value,ready[0],valid[-1],None if output is None else output[0]))
        if rst:
            coverage['reset_nonempty']+=int(bool(end_to_end));end_to_end.clear()
            for q in queues: q.clear()
            pending=None;continue
        push=bool(valid_in and ready[0]);pop=bool(valid[-1] and ready_out)
        if valid[-1]:
            if not end_to_end or output!=end_to_end[0]: raise InvalidInput('Stream oracle order invariant failed')
        if pop:
            token=end_to_end.popleft()
            if cycle<calibration_end: calibration_latencies.append(cycle-token[1])
        if push:
            end_to_end.append((value,cycle))
            if cycle<calibration_end: accepted.append(cycle)
        coverage['push']+=int(push);coverage['pop']+=int(pop);coverage['simultaneous']+=int(push and pop)
        coverage['input_stall']+=int(valid_in and not ready[0]);coverage['output_stall']+=int(valid[-1] and not ready_out)
        incoming=[(value,cycle)]+[q[0] if q else None for q in queues[:-1]]
        for i,q in enumerate(queues):
            downstream=ready_out if i==stages-1 else ready[i+1]
            if valid[i] and downstream: q.popleft()
            upstream=valid_in if i==0 else valid[i-1]
            if upstream and ready[i]: q.append(incoming[i])
            if len(q)>slots: raise InvalidInput('Stream oracle capacity invariant failed')
        pending=value if valid_in and not ready[0] else None
    if end_to_end or any(queues) or pending is not None: raise InvalidInput('Stream oracle failed to drain')
    intervals=[b-a for a,b in zip(accepted,accepted[1:])]
    if len(calibration_latencies)<64 or set(calibration_latencies)!={stages} or len(intervals)<64 or set(intervals)!={1}:
        raise InvalidInput('Stream no-stall cycle contract failed')
    calibration=dict(minimum_latency_cycles=stages,initiation_interval=1,latency_samples=len(calibration_latencies),capacity=slots*stages,architecture=architecture)
    return vectors,coverage,calibration
