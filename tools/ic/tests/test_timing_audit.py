"""Adversarial report-consistency controls; fixtures are explicitly synthetic."""
from pathlib import Path
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.exploration_common import fingerprint
from ic_core.tools.synth.timing_audit import CHECKS,parse_checks
from test_physical_analysis import record
from ic_core.tools.synth.clocked import DSP48E1_ACTIVE_REGS,DSP48E1_PROPERTIES


def fixture(store,tmp_path,monkeypatch):
    source=tmp_path/'source.sv';source.write_text('module dut; endmodule')
    sha=fingerprint([source],'dut')
    r=record('baseline','abcdef');r.update(source_sha256=sha,tool='vivado',stage='routed_clocked_ooc')
    r['evidence'].update(coverage='abcdef/coverage',constraint_checks='abcdef/checks')
    texts=dict(metrics='version=1\nbuild=fixture1\nslack_ns=0\nrequirement_ns=5\n',
        timing='| Design : dut\n| Design State : Routed\nSlack (MET) : 0.000ns\nPath Type: Setup\nPath Group: ic_clock\n',
        coverage=f'source_sha256={sha}\ntop=dut\nscope=single_clock_register_to_register_ooc\nclock_count=1\nclock_period_ns=5\nregister_count=20\nclocked_register_count=20\nlatch_count=0\nsetup_path_count=1\nio_delays_constrained=0\n',
        checks='| Design : dut\n'+'\n'.join(f'{i}. checking {name} (0)' for i,name in enumerate(CHECKS,1))+'\n'+'\n'.join(f'{i}. checking {name} (0)\nThere are 0 reported violations.\n' for i,name in enumerate(CHECKS,1)))
    paths={}
    for name,text in texts.items():
        path=tmp_path/name;path.write_text(text);paths['abcdef/'+name]=path
    monkeypatch.setattr(store,'resolve',lambda handle:paths[handle])
    return dict(files=[str(source)],top='dut',record=r),paths


def test_complete_coverage_has_explicit_scope(store,tmp_path,monkeypatch):
    p,_=fixture(store,tmp_path,monkeypatch)
    r=dispatch('timing_constraint_audit',p,store=store)
    assert r['ok'] and r['data']['requested_setup_met']
    assert 'hold-time closure' in r['data']['excluded']
    assert len(r['data']['report_sha256'])==4


@pytest.mark.parametrize('field,value',[('clock_count',2),('clocked_register_count',19),('latch_count',1),('setup_path_count',0)])
def test_bad_internal_coverage_fails(store,tmp_path,monkeypatch,field,value):
    p,paths=fixture(store,tmp_path,monkeypatch)
    file=paths['abcdef/coverage'];lines=file.read_text().splitlines()
    file.write_text('\n'.join(f'{field}={value}' if line.startswith(field+'=') else line for line in lines))
    r=dispatch('timing_constraint_audit',p,store=store)
    assert not r['ok'] and r['data']['reasons']


@pytest.mark.parametrize('mutation',['missing','mixed_run','source','period','slack','build','report','report_body','duplicate','clock_check'])
def test_incomplete_or_mismatched_evidence_rejected(store,tmp_path,monkeypatch,mutation):
    p,paths=fixture(store,tmp_path,monkeypatch)
    if mutation=='missing': del p['record']['evidence']['coverage']
    elif mutation=='mixed_run': p['record']['evidence']['coverage']='bbbbbb/coverage'
    elif mutation=='source': Path(p['files'][0]).write_text('module changed;endmodule')
    elif mutation in ('period','duplicate'):
        f=paths['abcdef/coverage'];f.write_text(f.read_text().replace('clock_period_ns=5','clock_period_ns=6') if mutation=='period' else f.read_text()+'clock_count=1\n')
    elif mutation in ('slack','build'):
        f=paths['abcdef/metrics'];f.write_text(f.read_text().replace('slack_ns=0','slack_ns=1') if mutation=='slack' else f.read_text().replace('fixture1','wrong'))
    elif mutation=='report': paths['abcdef/timing'].write_text('incomplete report')
    else:
        f=paths['abcdef/checks'];s=f.read_text()
        if mutation=='report_body': s=s.replace('There are 0','There are 1',1)
        else: s=s.replace('1. checking no_clock (0)','removed',1)
        f.write_text(s)
    with pytest.raises(InvalidInput): dispatch('timing_constraint_audit',p,store=store)


def test_negative_slack_is_not_requested_timing_closure(store,tmp_path,monkeypatch):
    p,paths=fixture(store,tmp_path,monkeypatch)
    p['record']['metrics'].update(slack_ns=-1,critical_period_ns=6,estimated_fmax_mhz=1000/6,throughput_mtransactions_s=1000/6)
    f=paths['abcdef/metrics'];f.write_text(f.read_text().replace('slack_ns=0','slack_ns=-1'))
    f=paths['abcdef/timing'];f.write_text(f.read_text().replace('Slack (MET) : 0.000ns','Slack (VIOLATED) : -1.000ns'))
    r=dispatch('timing_constraint_audit',p,store=store)
    assert r['ok'] and not r['data']['requested_setup_met']


def test_nonzero_check_summary_fails(store,tmp_path,monkeypatch):
    p,paths=fixture(store,tmp_path,monkeypatch)
    f=paths['abcdef/checks'];s=f.read_text().replace('checking loops (0)','checking loops (1)');f.write_text(s)
    r=dispatch('timing_constraint_audit',p,store=store)
    assert not r['ok'] and r['data']['violations']['loops']==1


def dsp_fixture(store,tmp_path,monkeypatch):
    p,paths=fixture(store,tmp_path,monkeypatch)
    f=paths['abcdef/coverage']
    props=dict.fromkeys(DSP48E1_ACTIVE_REGS,'0')
    props.update(name='core/dsp',REF_NAME='DSP48E1',USE_DPORT='0',USE_MULT='NONE',ADREG='1',DREG='1')
    f.write_text(f.read_text().replace('register_count=20\nclocked','register_count=21\nclocked')+'unclocked_cell_count=1\n'+''.join(f'unclocked.0.{k}={v}\n' for k,v in props.items()))
    return p,f


def test_only_unused_dsp_registers_are_excluded(store,tmp_path,monkeypatch):
    p,_=dsp_fixture(store,tmp_path,monkeypatch)
    r=dispatch('timing_constraint_audit',p,store=store)
    assert r['ok'] and len(r['data']['inactive_dsp48e1_cells'])==1
    assert r['data']['coverage']['register_count']==21
    assert r['data']['coverage']['clocked_register_count']==20


@pytest.mark.parametrize('prop,value',[(p,'1') for p in DSP48E1_ACTIVE_REGS]+[('USE_DPORT','1'),('USE_MULT','MULTIPLY'),('DREG','unknown')])
def test_active_or_unknown_dsp_storage_cannot_be_excluded(store,tmp_path,monkeypatch,prop,value):
    p,f=dsp_fixture(store,tmp_path,monkeypatch)
    lines=f.read_text().splitlines()
    f.write_text('\n'.join(f'unclocked.0.{prop}={value}' if line.startswith(f'unclocked.0.{prop}=') else line for line in lines))
    r=dispatch('timing_constraint_audit',p,store=store)
    assert not r['ok'] and not r['data']['inactive_dsp48e1_cells']


@pytest.mark.parametrize('mutation',['missing_property','wrong_count','extra_cell','duplicate_name'])
def test_incomplete_dsp_property_evidence_rejected(store,tmp_path,monkeypatch,mutation):
    p,f=dsp_fixture(store,tmp_path,monkeypatch);text=f.read_text()
    if mutation=='missing_property':text=text.replace('unclocked.0.PREG=0\n','')
    elif mutation=='wrong_count':text=text.replace('unclocked_cell_count=1','unclocked_cell_count=0')
    elif mutation=='extra_cell':text+='unclocked.1.name=extra\n'
    else:
        text=text.replace('register_count=21','register_count=22').replace('unclocked_cell_count=1','unclocked_cell_count=2')
        text+='\n'.join(line.replace('unclocked.0.','unclocked.1.') for line in text.splitlines() if line.startswith('unclocked.0.'))+'\n'
    f.write_text(text)
    with pytest.raises(InvalidInput):dispatch('timing_constraint_audit',p,store=store)
