"""Matrix oracles, protocol traces and deliberately broken PE dataflows."""
from pathlib import Path
import shutil
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.debug.matrix import matrix_result, matrix_trace, pack


def check(store,data,**extra):
    payload=dict(files=data['files'],top=data['top'],width=data['width'],size=data['size'],operation='matrix',
                 latency_cycles=data['latency_cycles'],expected_ii=data['initiation_interval'],random_cycles=8192)
    payload.update(extra)
    return dispatch('latency_throughput',payload,store=store)


def test_matrix_oracle_known_result_and_signed_full_precision():
    assert matrix_result(pack([1,2,3,4,5,6,7,8],16),16,2)[0]==pack([19,22,43,50],33)
    assert matrix_result(pack([-128]*8,8),8,2)[0]==pack([32768]*4,17)
    assert matrix_result(pack([-128]*4+[127]*4,8),8,2)[0]==pack([-32512]*4,17)


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('width,size',[(16,2),(32,2),(8,4),(16,3)])
@pytest.mark.parametrize('architecture',['parallel','systolic'])
def test_signed_matrix_dataflow_and_observations(store,width,size,architecture):
    data=dispatch('synth_systolic_tile',dict(width=width,size=size,architecture=architecture),store=store)['data']
    for fixture in (False,True):
        result=check(store,data,files=data['timing_files'] if fixture else data['files'],
                     top=data['timing_top'] if fixture else data['top'],timing_fixture=fixture,
                     random_cycles=8192 if size==2 else 512)
        assert result['ok'],result
        assert result['data']['latency_range']==[data['latency_cycles']]*2
        assert result['data']['steady_state_acceptance_intervals']==[data['initiation_interval']]


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('mutation',['forward','skew','signed','accumulate','reset','hold','latency','ii','fixture'])
def test_broken_wavefront_or_protocol_is_rejected(store,tmp_path,mutation):
    data=dispatch('synth_systolic_tile',dict(width=16,size=2),store=store)['data']
    source=Path(data['files'][0]).read_text();extra=dict(random_cycles=512)
    if mutation=='forward': source=source.replace('a_0_0 * above_1','a_1_0 * above_1')
    if mutation=='skew': source=source.replace('(phase==1) ?','(phase==0) ?')
    if mutation=='signed': source=source.replace('reg signed','reg').replace('wire signed','wire')
    if mutation=='accumulate': source=source.replace('sum_0_0 + ','')
    if mutation=='reset': source=source.replace('if(rst) begin busy<=0;valid_q<=0','if(rst) begin busy<=0;valid_q<=1')
    if mutation=='hold': source=source.replace('if(valid_q && ready_out)','if(valid_q)')
    if mutation=='latency': extra['latency_cycles']=4
    if mutation=='ii': extra['expected_ii']=1
    altered=tmp_path/'bad.sv';altered.write_text(source)
    extra['files']=[str(altered)]
    if mutation=='fixture':
        altered.write_text(Path(data['timing_files'][1]).read_text().replace('valid_out<=core_valid;','valid_out<=valid_q;'))
        extra.update(files=[data['files'][0],str(altered)],top=data['timing_top'],timing_fixture=True)
    out=check(store,data,**extra)
    assert not out['ok'],out
    assert out['data'].get('verified_initiation_interval') is None


def test_trace_holds_pending_matrix_and_drains():
    trace,coverage,accepted,latencies=matrix_trace(32,2,5,8192,928)
    assert all(coverage.values()) and set(latencies)=={5}
    assert set(b-a for a,b in zip(accepted,accepted[1:]))=={5}
    assert trace==matrix_trace(32,2,5,8192,928)[0]
    for before,after in zip(trace,trace[1:]):
        if not before[0] and before[1] and not before[4] and not after[0]:
            assert after[1] and before[3]==after[3]


@pytest.mark.parametrize('payload',[dict(size=1),dict(size=5),dict(width=33)])
def test_bounds(store,payload):
    with pytest.raises(InvalidInput): dispatch('synth_systolic_tile',payload,store=store)
