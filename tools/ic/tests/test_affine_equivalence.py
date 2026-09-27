"""Conservative netlist algebra, with nonlinear/state/unknown rejection controls."""
from copy import deepcopy
import shutil
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.debug.affine import affine_rows,evaluate


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
