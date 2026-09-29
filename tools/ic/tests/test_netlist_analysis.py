"""Digest-bound netlist graph and memory-classification tests."""
import hashlib
import json
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput


def fixture(tmp_path):
    module=dict(ports={'x':{'direction':'input','bits':[2]},'y':{'direction':'output','bits':[3]}},netnames={'signal':{'bits':[2]}},cells={
        'logic':dict(type='$and',connections={'A':[2],'B':[2],'Y':[3]},port_directions={'A':'input','B':'input','Y':'output'},parameters={}),
        'ram':dict(type='$mem_v2',connections={},port_directions={},parameters={'WIDTH':'00010000','SIZE':'01000000','RD_PORTS':'1','WR_PORTS':'1'}),
        'block':dict(type='RAMB18E1',connections={},port_directions={},parameters={}),
        'distributed':dict(type='RAM64M',connections={},port_directions={},parameters={}),
        'flop':dict(type='FDRE',connections={'D':['0'],'Q':[4],'C':[2]},port_directions={'D':'input','Q':'output','C':'input'},parameters={})})
    path=tmp_path/'netlist.json';path.write_text(json.dumps({'modules':{'dut':module}}))
    return dict(netlist_file=str(path),top='dut',expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_pin_fanout_includes_control_and_excludes_constants(store,tmp_path):
    p=fixture(tmp_path)
    result=dispatch('fanout_analysis',dict(p,threshold=2),store=store)['data']
    assert result['total_flagged']==1
    row=result['nets'][0]
    assert row['sink_pins']==3 and row['bit']==2
    assert row['drivers']==['port:x[0]']
    assert 'flop.C[0]' in row['sample_sinks']


def test_memory_classes_are_not_conflated(store,tmp_path):
    data=dispatch('memory_inference',fixture(tmp_path),store=store)['data']
    assert data['logical_memory_bits']==1024
    assert data['physical_primitives']==dict(bram18_primitives=1,distributed_ram_primitives=1,register_output_bits=1)
    assert data['logical_memories'][0]['depth']==64


@pytest.mark.parametrize('op',['fanout_analysis','memory_inference'])
def test_modified_netlist_is_rejected(store,tmp_path,op):
    p=fixture(tmp_path)
    from pathlib import Path
    with Path(p['netlist_file']).open('a') as stream: stream.write(' ')
    with pytest.raises(InvalidInput,match='digest mismatch'): dispatch(op,p,store=store)


def test_malformed_json_rejected(store,tmp_path):
    path=tmp_path/'bad.json';path.write_text('{broken')
    p=dict(netlist_file=str(path),top='dut',expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(InvalidInput,match='Malformed'): dispatch('memory_inference',p,store=store)


import shutil
@pytest.mark.skipif(not shutil.which('yosys'),reason='Local Yosys unavailable')
@pytest.mark.parametrize('mapping',['generic','xc7'])
def test_real_fifo_netlist(store,mapping):
    data=dispatch('synth_fifo',dict(width=8,depth=16,architecture='circular'),store=store)['data']
    profile=dispatch('netlist_profile',dict(files=data['files'],top='fifo_dut',mapping=mapping),store=store)
    assert profile['ok'],profile
    parsed=dispatch('memory_inference',dict(netlist_file=profile['data']['netlist_file'],top='fifo_dut',expected_sha256=profile['data']['netlist_sha256']),store=store)['data']
    if mapping=='generic': assert parsed['logical_memory_bits']==128
    else:
        resources=parsed['physical_primitives']
        assert (sum(resources.get(k,0) for k in ('distributed_ram_primitives','bram18_primitives','bram36_primitives'))>0
                or resources.get('register_output_bits',0)>=128)


def test_cone_has_logic_depth_and_state_boundaries(store,tmp_path):
    p=fixture(tmp_path)
    result=dispatch('critical_cone',dict(p,endpoints=['y']),store=store)['data']
    assert result['logic_levels']==1 and result['combinational_cells']==1
    assert result['longest_structural_path']==['logic']
    with pytest.raises(InvalidInput,match='Unknown'): dispatch('critical_cone',dict(p,endpoints=['missing']),store=store)


def test_combinational_cycle_is_not_a_timing_estimate(store,tmp_path):
    p=fixture(tmp_path)
    from pathlib import Path
    path=Path(p['netlist_file']);document=json.loads(path.read_text())
    document['modules']['dut']['cells']['logic']['connections']['A']=[3]
    path.write_text(json.dumps(document));p['expected_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(InvalidInput,match='Combinational cycle'): dispatch('critical_cone',dict(p,endpoints=['y']),store=store)


def test_state_is_a_cone_boundary(store,tmp_path):
    p=fixture(tmp_path)
    from pathlib import Path
    path=Path(p['netlist_file']);document=json.loads(path.read_text())
    document['modules']['dut']['cells']['logic']['connections']['A']=[4]
    path.write_text(json.dumps(document));p['expected_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dispatch('critical_cone',p,store=store)['data']
    assert result['logic_levels']==1 and result['boundary_cells']==['flop']
