"""Conservative netlist algebra, with nonlinear/state/unknown rejection controls."""
from copy import deepcopy
from pathlib import Path
import shutil
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.debug.affine import affine_rows,evaluate
from ic_core.tools.exploration_common import fingerprint


def module():
    return dict(ports={'x':dict(direction='input',bits=[2,3,4]),'y':dict(direction='output',bits=[7,8])},cells={
        'op':dict(type='$xor',port_directions=dict(A='input',B='input',Y='output'),connections=dict(A=[2,3],B=[4,'1'],Y=[7,8]),parameters={})})


def test_affine_xor_and_bias_are_exact():
    rows=affine_rows(module(),3,2)
    assert rows==[5,10]
    for value in range(8):
        assert evaluate(rows,value,3)==(((value&1)^((value>>2)&1))|((((value>>1)&1)^1)<<1))


@pytest.mark.parametrize('kind',['$and','$or'])
def test_variable_products_are_not_a_linear_proof(kind):
    data=module();data['cells']['op']['type']=kind
    with pytest.raises(InvalidInput,match='Nonlinear'): affine_rows(data,3,2)


@pytest.mark.parametrize('is_signed,expected',[(False,[1,0]),(True,[1,2])])
def test_constant_masks_use_correct_signed_extension(is_signed,expected):
    data=module();cell=data['cells']['op'];cell['type']='$and';cell['connections']['B']=['1']
    cell['parameters']=dict(A_SIGNED=int(is_signed),B_SIGNED=int(is_signed))
    assert affine_rows(data,3,2)==expected


@pytest.mark.parametrize('fault,match',[
    ('cycle','Cycle'),('undriven','Undriven'),('unknown','unknown'),('state','stateful'),
    ('multiple','Multiple'),('interface','interface'),('port','complete'),
])
def test_invalid_netlist_rejected(fault,match):
    data=module();cell=data['cells']['op']
    if fault=='cycle': cell['connections']['A']=[7,8]
    elif fault=='undriven': cell['connections']['A']=[91,92]
    elif fault=='unknown': cell['connections']['B']=['x','1']
    elif fault=='state': cell['type']='$dff'
    elif fault=='multiple': data['cells']['duplicate']=deepcopy(cell)
    elif fault=='interface': data['ports']['x']['bits']=[2,3]
    elif fault=='port': data['ports']['clk']=dict(direction='input',bits=[99])
    with pytest.raises(InvalidInput,match=match): affine_rows(data,3,2)


def test_reduction_and_inversion_with_reversed_dependency_order():
    data=module()
    data['cells']={
        'first':dict(type='$not',connections=dict(A=[9,9],Y=[7,8]),port_directions=dict(A='input',Y='output'),parameters={}),
        'last':dict(type='$reduce_xor',connections=dict(A=[2,3,4],Y=[9]),port_directions=dict(A='input',Y='output'),parameters={})}
    assert affine_rows(data,3,2)==[15,15]


@pytest.mark.skipif(not shutil.which('yosys'),reason='Local Yosys unavailable')
@pytest.mark.parametrize('wrong',[False,True])
def test_actual_source_bound_affine_check(store,wrong):
    a=dispatch('synth_crc_parallel',dict(data_width=64,architecture='unrolled'),store=store)['data']
    b=dispatch('synth_crc_parallel',dict(data_width=64,architecture='shared',polynomial=0x1EDC6F41 if wrong else 0x04C11DB7),store=store)['data']
    result=dispatch('gf2_equivalence',dict(reference_files=[a['core_path']],candidate_files=[b['core_path']],input_width=96,output_width=32),store=store)
    assert result['ok'] is (not wrong)
    assert (result['data']['counterexample'] is not None) is wrong
    if wrong:
        witness=result['data']['counterexample']
        assert witness['reference_y']!=witness['candidate_y']


@pytest.mark.parametrize('failed_role',[0,1])
@pytest.mark.parametrize('fault',['failure','missing_interface','partial_semantics','wrong_hash','missing_proof'])
def test_source_guards_precede_affine_simplification(store,tmp_path,monkeypatch,failed_role,fault):
    import hashlib
    import json
    from ic_core.tools.debug import affine
    files=[]
    for role in ('reference','candidate'):
        path=tmp_path/(role+'.sv')
        path.write_text('module dut(input x,output y); assign y=x; endmodule')
        files.append(path)
    netlist=tmp_path/'netlist.json'
    netlist.write_text(json.dumps({'modules':{'dut':{'ports':{'x':{'direction':'input','bits':[2]},'y':{'direction':'output','bits':[2]}},'cells':{}}}}))
    sha=hashlib.sha256(netlist.read_bytes()).hexdigest()
    calls=[];guards=[]
    def fake_dispatch(op,params,**kwargs):
        calls.append(op)
        if op=='yosys_equivalence':
            assert params['reference_files']==params['candidate_files']
            source_sha=fingerprint([Path(params['reference_files'][0])],'dut')
            data=dict(proved=True,interface_complete=True,abstraction_defined=True,reference_sha256=source_sha,candidate_sha256=source_sha)
            result=dict(ok=True,data=data)
            if len(guards)==failed_role:
                if fault=='failure':result['ok']=False
                elif fault=='missing_interface':data.pop('interface_complete')
                elif fault=='partial_semantics':data['abstraction_defined']=False
                elif fault=='wrong_hash':data['candidate_sha256']='0'*64
                elif fault=='missing_proof':data.pop('proved')
            guards.append(result)
            return result
        assert op=='netlist_profile'
        return dict(ok=True,data=dict(source_sha256=fingerprint([files[0]],'dut'),netlist_file=str(netlist),netlist_sha256=sha))
    monkeypatch.setattr(affine,'dispatch',fake_dispatch)
    result=dispatch('gf2_equivalence',dict(reference_files=[str(files[0])],candidate_files=[str(files[1])],input_width=1,output_width=1),store=store)
    assert not result['ok'] and not result['data']['proved']
    assert result['data']['source_guards']==guards
    assert calls==(['yosys_equivalence'] if failed_role==0 else ['yosys_equivalence','netlist_profile','yosys_equivalence'])


def test_total_binary_pmux_requires_complete_exclusive_decode():
    from ic_core.tools.debug.formal import total_binary_module
    cells={}
    selectors=[]
    for value in range(4):
        bit=20+value;selectors.append(bit)
        cells[f'eq{value}']=dict(type='$eq',connections=dict(A=[2,3],B=[str(value&1),str(value>>1)],Y=[bit]))
    cells['mux']=dict(type='$pmux',connections=dict(A=['x'],B=[4,5,6,7],S=selectors,Y=[8]))
    module=dict(cells=cells,netnames={'all':{'bits':[2,3,*selectors,8]}})
    assert total_binary_module(module)
    cells['eq3']['connections']['B']=['0','0']
    assert not total_binary_module(module)
    cells['eq3']['connections']['B']=['1','1'];cells.pop('eq2')
    assert not total_binary_module(module)
