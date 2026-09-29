"""Adversarial small evidence stores; synthetic metric records are test fixtures."""
import json
from pathlib import Path
import pytest
from ic_core import dispatch
from ic_core.errors import InvalidInput
from ic_core.tools.exploration_common import fingerprint
from test_physical_analysis import record


def write(path,value):
    path.write_text(json.dumps(value));return path


def fixture(tmp_path,store):
    root=tmp_path/'experiments';root.mkdir();folder=root/'exp-tool-50-test';folder.mkdir()
    (folder/'README.md').write_text('Bounded test fixture')
    inv=write(tmp_path/'inventory.json',[dict(id=50,operation='acceptance_audit',direct_ppa=True,purpose='Audit evidence',paper='https://example.test/paper',tests=[str(Path(__file__).resolve())])])
    evidence=tmp_path/'evidence';evidence.mkdir();checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    config=dict(name='fifo',generator='synth_fifo',baseline=dict(width=16,depth=64,architecture='shift'),candidate=dict(width=16,depth=64,architecture='circular'),objective='area')
    write(folder/'config.json',dict(cases=[config]))
    designs=[];checks=[]
    for role in ('baseline','candidate'):
        generated=dispatch('synth_fifo',config[role],store=store);designs.append(generated['data'])
        write(checkpoint/f'fifo-{role}-generate.json',dict(operation='synth_fifo',payload=config[role],result=generated))
        for key in ('source_sha256','timing_sha256'):
            checks.append(dict(ok=True,data=dict(passed=True,source_unchanged=True,source_sha256=generated['data'][key])))
    pairs=[]
    for i in range(3):
        pair=[]
        for role,d in zip(('baseline','candidate'),designs):
            row=record(role,f'{role}{i}',luts=100 if role=='baseline' else 70)
            row.update(source_sha256=d['timing_sha256'],tool='vivado',stage='routed_clocked_ooc',latency_cycles=3)
            pair.append(row)
        pairs.append(pair)
    case=dict(name='fifo',generator='synth_fifo',objective='area',status='measured',records=pairs,checks=checks)
    report=dict(ok=True,run_id='fixture',data=dict(checkpoint_directory=str(checkpoint),cases=[case]))
    path=write(evidence/'study.json',report)
    payload=dict(experiment_root=str(root),evidence_root=str(evidence),inventory=str(inv))
    return payload,report,path,folder,designs


def test_complete_small_data_is_not_whole_goal_completion(tmp_path,store):
    p,_,_,_,_=fixture(tmp_path,store)
    out=dispatch('acceptance_audit',p,store=store)['data']
    assert not out['automated_gates_passed'] and not out['completion_claim_supported']
    assert out['declared_cases']==1 and out['cases_with_verified_physical_pairs']==1
    assert out['all_case_geometric_benefit_lower_bound']==pytest.approx(100/70)
    assert out['qualifying_family_configurations']==dict(synth_fifo=1)


def test_audit_reports_repository_files_with_portable_paths(tmp_path,store):
    p,_,_,_,_=fixture(tmp_path,store)
    out=dispatch('acceptance_audit',p,store=store,cwd=tmp_path)['data']
    assert 'inventory.json' in out['input_sha256']
    assert 'experiments/exp-tool-50-test/config.json' in out['input_sha256']
    assert 'evidence/study.json' in out['input_sha256']
    assert out['cases'][0]['declarations']==['experiments/exp-tool-50-test/config.json']
    assert out['cases'][0]['observations'][0]['file']=='evidence/study.json'


def test_prior_acceptance_output_is_not_its_own_input(tmp_path,store):
    p,_,path,_,_=fixture(tmp_path,store)
    previous=dict(ok=False,data=dict(gates={},declared_cases=1,
        input_sha256={'old':'digest'},completion_claim_supported=False))
    write(path.parent/'acceptance-audit.json',previous)
    out=dispatch('acceptance_audit',p,store=store,cwd=tmp_path)['data']
    assert 'evidence/acceptance-audit.json' not in out['input_sha256']
    assert out['declared_cases']==1 and out['cases_with_verified_physical_pairs']==1


@pytest.mark.parametrize('mutation',['failed','missing','source','proof','cycle','duplicate','synthetic','objective'])
def test_unknown_or_invalid_evidence_cannot_supply_ratio(tmp_path,store,mutation):
    p,report,path,folder,designs=fixture(tmp_path,store);case=report['data']['cases'][0]
    if mutation=='failed': case['status']='measurement_failed'
    if mutation=='missing': report['data']['cases']=[]
    if mutation=='source': Path(designs[1]['files'][0]).write_text('changed')
    if mutation=='proof': case['checks'][1]['data']['passed']=False
    if mutation=='cycle': case['records'][0][1]['latency_cycles']=9
    if mutation=='duplicate': case['records'][1]=case['records'][0]
    if mutation=='synthetic': case['records'][0][1]['stage']='synthetic_fixture'
    if mutation=='objective': case['objective']='throughput'
    write(path,report)
    out=dispatch('acceptance_audit',p,store=store)['data']
    assert out['all_case_geometric_benefit_lower_bound'] is None
    assert not out['gates']['all_declared_cases_measured']
    assert out['declared_cases']==1


def test_unmeasured_declaration_remains_in_denominator(tmp_path,store):
    p,_,_,folder,_=fixture(tmp_path,store)
    config=json.loads((folder/'config.json').read_text());second=json.loads(json.dumps(config['cases'][0]))
    second['baseline']['width']=32;second['candidate']['width']=32;config['cases'].append(second)
    write(folder/'config.json',config)
    out=dispatch('acceptance_audit',p,store=store)['data']
    assert out['declared_cases']==2 and out['cases_with_verified_physical_pairs']==1
    assert out['all_case_geometric_benefit_lower_bound'] is None


def test_worst_complete_retry_prevents_cherry_picking(tmp_path,store):
    p,report,path,_,_=fixture(tmp_path,store)
    report['run_id']='regression'
    for pair in report['data']['cases'][0]['records']: pair[1]['metrics']['luts']=110
    write(path.parent/'retry.json',report)
    out=dispatch('acceptance_audit',p,store=store)['data']
    assert out['all_case_geometric_benefit_lower_bound']==pytest.approx(100/110)
    assert not out['cases'][0]['qualified'] and len(out['cases'][0]['observations'])==2


def test_duplicate_tools_rejected(tmp_path,store):
    p,*_=fixture(tmp_path,store);path=Path(p['inventory']);inv=json.loads(path.read_text());write(path,inv+inv)
    with pytest.raises(InvalidInput,match='Duplicate operation'): dispatch('acceptance_audit',p,store=store)


def test_malformed_paper_reference_fails_inventory_structure(tmp_path,store):
    p,*_=fixture(tmp_path,store);path=Path(p['inventory']);inventory=json.loads(path.read_text())
    inventory[0]['paper']='https://example.test/paper,';write(path,inventory)
    out=dispatch('acceptance_audit',p,store=store)['data']
    assert not out['gates']['inventory_structure']
    assert 'malformed primary paper' in out['tools'][0]['errors'][0]


def test_combinational_core_and_wrapper_hashes_are_distinct(store):
    from ic_core.tools.synth.acceptance import source_check
    designs=[dispatch('synth_prefix_adder',dict(width=16,architecture=a),store=store)['data'] for a in ('native','kogge_stone')]
    hashes=[fingerprint([Path(d['core_path'])],'dut') for d in designs]
    records=[]
    for role,d in zip(('baseline','candidate'),designs):
        r=record(role,role);r.update(source_sha256=d['source_sha256'],stage='routed_clocked_ooc',tool='vivado')
        records.append(r)
    checks=dict(reference_sha256=hashes[0],candidate_sha256=hashes[1],interface_complete=True,abstraction_defined=True)
    row=dict(simulation=dict(ok=True,data=dict(passed=True,**checks)),formal=dict(ok=True,data=dict(proved=True,**checks)),records=[records])
    assert source_check(row,designs)==[d['source_sha256'] for d in designs]
    row['formal']['data']['proved']=False
    with pytest.raises(InvalidInput,match='correctness'): source_check(row,designs)


@pytest.mark.parametrize('field',['interface_complete','abstraction_defined'])
def test_combinational_proof_requires_complete_defined_interface(store,field):
    from ic_core.tools.synth.acceptance import source_check
    designs=[dispatch('synth_prefix_adder',dict(width=16,architecture=a),store=store)['data'] for a in ('native','kogge_stone')]
    hashes=[fingerprint([Path(d['core_path'])],'dut') for d in designs]
    records=[]
    for role,d in zip(('baseline','candidate'),designs):
        r=record(role,role);r.update(source_sha256=d['source_sha256'],stage='routed_clocked_ooc',tool='vivado');records.append(r)
    proof=dict(proved=True,reference_sha256=hashes[0],candidate_sha256=hashes[1],interface_complete=True,abstraction_defined=True)
    proof[field]=False
    row=dict(simulation=dict(ok=True,data=dict(passed=True,reference_sha256=hashes[0],candidate_sha256=hashes[1])),formal=dict(ok=True,run_id='proof',data=proof),records=[records])
    with pytest.raises(InvalidInput,match='complete-interface'): source_check(row,designs)


def test_historical_interface_supplement_is_source_bound(store,tmp_path):
    from ic_core.tools.synth.acceptance import correctness_supplements,source_check
    designs=[dispatch('synth_prefix_adder',dict(width=16,architecture=a),store=store)['data'] for a in ('native','kogge_stone')]
    hashes=[fingerprint([Path(d['core_path'])],'dut') for d in designs]
    records=[]
    for role,d in zip(('baseline','candidate'),designs):
        r=record(role,role);r.update(source_sha256=d['source_sha256'],stage='routed_clocked_ooc',tool='vivado');records.append(r)
    checks=dict(reference_sha256=hashes[0],candidate_sha256=hashes[1])
    row=dict(simulation=dict(ok=True,data=dict(passed=True,**checks)),formal=dict(ok=True,run_id='old-proof',data=dict(proved=True,**checks)),records=[records])
    audit=write(tmp_path/'interfaces.json',dict(scope='Interface and total-binary source-netlist re-elaboration test fixture',rows=[dict(role=role,source_sha256=sha,interface_complete=True,abstraction_defined=True) for role,sha in zip(('reference','candidate'),hashes)],passed=True))
    supplements=correctness_supplements([str(audit)],tmp_path,{})
    assert source_check(row,designs,supplements)==[d['source_sha256'] for d in designs]
    document=json.loads(audit.read_text());document['rows'][0]['source_sha256']='0'*64;write(audit,document)
    with pytest.raises(InvalidInput,match='complete-interface'):
        source_check(row,designs,correctness_supplements([str(audit)],tmp_path,{}))
    document['rows'][0]['source_sha256']=hashes[0];document['rows'][0]['abstraction_defined']=False;write(audit,document)
    with pytest.raises(InvalidInput,match='failed source audit'):
        source_check(row,designs,correctness_supplements([str(audit)],tmp_path,{}))


def test_registry_method_with_unindented_embedded_rtl(tmp_path,store):
    p,*_=fixture(tmp_path,store);path=Path(p['inventory']);inv=json.loads(path.read_text())
    inv[0]['operation']='comb_check';write(path,inv)
    out=dispatch('acceptance_audit',p,store=store)['data']
    assert out['tools'][0]['structurally_present']


def test_unrecognized_legacy_declaration_blocks_aggregate(tmp_path,store):
    p,_,_,folder,_=fixture(tmp_path,store)
    write(folder/'unknown.json',dict(widths=[16,32],architectures=['a','b']))
    out=dispatch('acceptance_audit',p,store=store)['data']
    assert out['unsupported_declarations'] and out['all_case_geometric_benefit_lower_bound'] is None


def test_new_coverage_cannot_be_ignored_or_replaced_by_success_flag(tmp_path,store,monkeypatch):
    from ic_core.tools.synth import acceptance
    from ic_core.tools.synth.exploration import Result
    p,report,path,_,_=fixture(tmp_path,store)
    for pair in report['data']['cases'][0]['records']:
        for r in pair: r['evidence'].update(coverage='abcdef/coverage',constraint_checks='abcdef/checks')
    write(path,report)
    monkeypatch.setattr(acceptance.Audit,'timing_constraint_audit',lambda self,p,ctx:Result(ok=False,data=dict(reasons=['unclocked domain'])))
    out=dispatch('acceptance_audit',p,store=store)['data']
    assert out['all_case_geometric_benefit_lower_bound'] is None
    assert 'unclocked domain' in out['cases'][0]['observations'][0]['error']


def test_historical_coverage_is_explicitly_unaudited(tmp_path,store):
    p,*_=fixture(tmp_path,store)
    out=dispatch('acceptance_audit',p,store=store)['data']
    coverage=out['cases'][0]['observations'][0]['timing_coverage']
    assert coverage==dict(audited_records=0,historical_unaudited_records=6)
