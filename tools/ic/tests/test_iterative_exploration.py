"""Independent transaction traces and adversarial cycle contracts."""
from pathlib import Path
import shutil
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.debug.cycles import arithmetic_trace


def check(store, data, **extra):
    payload = dict(files=data['files'], width=data['width'], operation=data['operation'],
                   latency_cycles=data['latency_cycles'], expected_ii=data['initiation_interval'], random_cycles=8192)
    payload.update(extra)
    return dispatch('latency_throughput', payload, store=store)


@pytest.mark.skipif(not shutil.which('iverilog'), reason='Icarus unavailable')
@pytest.mark.parametrize('width', [16, 32])
@pytest.mark.parametrize('architecture', ['parallel', 'serial', 'radix4'])
@pytest.mark.parametrize('op', ['synth_serial_multiplier', 'synth_divider'])
def test_arithmetic_and_registered_observations(store, width, architecture, op):
    data = dispatch(op, dict(width=width, architecture=architecture), store=store)['data']
    for fixture in (False, True):
        result = check(store, data, files=data['timing_files'] if fixture else data['files'],
                       top=data['timing_top'] if fixture else data['top'], timing_fixture=fixture)
        assert result['ok'], result
        assert result['data']['source_sha256'] == data['timing_sha256' if fixture else 'source_sha256']
        assert result['data']['latency_range'] == [data['latency_cycles']]*2
        assert result['data']['steady_state_acceptance_intervals'] == [data['initiation_interval']]


@pytest.mark.parametrize('operation', ['multiply', 'divide'])
def test_trace_is_deterministic_and_holds_inputs(operation):
    trace, coverage, accepted, latencies = arithmetic_trace(32, operation, 33, 8192, 928)
    assert all(coverage.values())
    assert trace == arithmetic_trace(32, operation, 33, 8192, 928)[0]
    assert set(latencies) == {33}
    assert set(b-a for a,b in zip(accepted, accepted[1:])) == {33}
    for before, after in zip(trace, trace[1:]):
        if not before[0] and before[1] and not before[4] and not after[0]:
            assert after[1] and before[3] == after[3]


@pytest.mark.skipif(not shutil.which('iverilog'), reason='Icarus unavailable')
@pytest.mark.parametrize('mutation', ['result', 'reset', 'hold', 'latency', 'ii', 'division_zero'])
def test_cycle_checker_rejects_mutations(store, tmp_path, mutation):
    op = 'synth_divider' if mutation == 'division_zero' else 'synth_serial_multiplier'
    data = dispatch(op, dict(width=16, architecture='parallel' if mutation=='division_zero' else 'serial'), store=store)['data']
    source = Path(data['files'][0]).read_text()
    extra = dict(random_cycles=512)
    if mutation == 'result': source = source.replace('data_out<=acc0;', "data_out<=acc0 ^ 32'd1;")
    if mutation == 'reset': source = source.replace('busy<=0;valid_q<=0;data_out<=0;', 'busy<=0;valid_q<=1;data_out<=0;')
    if mutation == 'hold': source = source.replace('if(valid_q && ready_out)', 'if(valid_q)')
    if mutation == 'division_zero': source = source.replace("{a,16'd65535}", "{a,16'd0}")
    if mutation == 'latency': extra['latency_cycles'] = data['latency_cycles']-1
    if mutation == 'ii': extra['expected_ii'] = data['initiation_interval']-1
    altered = tmp_path/'mutated.sv'; altered.write_text(source)
    result = check(store, data, files=[str(altered)], **extra)
    assert not result['ok'], result
    assert result['data']['verified_latency_cycles'] is None
    if mutation != 'ii': assert result['data']['first_mismatch'] is not None


def test_radix4_requires_whole_two_bit_steps(store):
    with pytest.raises(InvalidInput):
        dispatch('synth_divider', dict(width=17, architecture='radix4'), store=store)


@pytest.mark.skipif(not shutil.which('iverilog'), reason='Icarus unavailable')
@pytest.mark.parametrize('width,architecture', [(8,'parallel'),(17,'serial'),(64,'radix4')])
@pytest.mark.parametrize('op', ['synth_serial_multiplier','synth_divider'])
def test_bounded_width_edges(store,width,architecture,op):
    data=dispatch(op,dict(width=width,architecture=architecture),store=store)['data']
    result=check(store,data,random_cycles=512)
    assert result['ok'],result


@pytest.mark.skipif(not shutil.which('iverilog'), reason='Icarus unavailable')
def test_wrong_fixture_observation_delay_is_rejected(store,tmp_path):
    data=dispatch('synth_serial_multiplier',dict(width=32,architecture='radix4'),store=store)['data']
    source=Path(data['timing_files'][1]).read_text().replace('valid_out<=core_valid;', 'valid_out<=valid_q;')
    altered=tmp_path/'bad_fixture.sv';altered.write_text(source)
    result=check(store,data,files=[data['files'][0],str(altered)],top=data['timing_top'],timing_fixture=True,random_cycles=512)
    assert not result['ok'] and result['data']['first_mismatch'] is not None
