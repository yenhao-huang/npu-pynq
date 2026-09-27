"""FIFO protocol checks and negative controls for temporal behavior."""
from pathlib import Path
import shutil
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.debug.sequential import fifo_vectors


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('width,depth',[(16,64),(32,128)])
@pytest.mark.parametrize('architecture',['shift','circular'])
def test_fifo_transaction_model(store,width,depth,architecture):
    generated=dispatch('synth_fifo',dict(width=width,depth=depth,architecture=architecture),store=store)['data']
    result=dispatch('sequential_scoreboard',dict(files=generated['files'],width=width,depth=depth,random_cycles=8192),store=store)
    assert result['ok'], result
    assert all(result['data']['coverage'].values())
    assert result['data']['source_sha256']==generated['source_sha256']


@pytest.mark.skipif(not shutil.which('iverilog'),reason='Icarus unavailable')
@pytest.mark.parametrize('mutation', ['order','reset','capacity'])
def test_scoreboard_rejects_protocol_bugs(store,tmp_path,mutation):
    data=dispatch('synth_fifo',dict(width=16,depth=64),store=store)['data']
    source=Path(data['files'][0]).read_text()
    if mutation=='order': source=source.replace("read_ptr<=read_ptr+1'b1", "read_ptr<=read_ptr+2'd2")
    if mutation=='reset': source=source.replace('if(rst) count<=0;', 'if(rst) count<=1;')
    if mutation=='capacity': source=source.replace("count < 7'd64", "count < 7'd63")
    altered=tmp_path/'bad_fifo.sv';altered.write_text(source)
    result=dispatch('sequential_scoreboard',dict(files=[str(altered)],width=16,depth=64,random_cycles=512),store=store)
    assert not result['ok'] and result['data']['first_mismatch'] is not None


def test_stimulus_holds_backpressured_data():
    vectors,coverage=fifo_vectors(16,64,8192,928)
    for before,after in zip(vectors,vectors[1:]):
        if not before[0] and before[1] and not before[4] and not after[0]:
            assert after[1] and before[3]==after[3]
    assert all(coverage.values())
    assert vectors==fifo_vectors(16,64,8192,928)[0]


def test_fifo_rejects_non_power_capacity(store):
    with pytest.raises(InvalidInput): dispatch('synth_fifo',dict(depth=63),store=store)
