"""Fail-closed inventory and source-bound whole-study PPA acceptance audit."""
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
from urllib.parse import urlsplit
from pydantic import Field
from ...errors import InvalidInput
from ...registry import Op, backend, find_op
from ..exploration_common import ExplorationInput, fingerprint
from . import CATEGORY
from .exploration import Result
from .physical_analysis import ClockPair, PhysicalAnalysis, RepeatsIn
from .timing_audit import Audit, AuditIn


class AcceptanceIn(ExplorationInput):
    backend: str | None = Field(default='acceptance',description='Whole-study evidence inventory and quantitative acceptance audit.')
    experiment_root: str = Field(description='Root containing every modular exp-tool-* declaration, including unsuccessful cases.')
    evidence_root: str = Field(description='Root containing JSON study evidence, recursively inspected without selecting only successes.')
    inventory: str = Field(description='JSON inventory of operation IDs, names, PPA purposes and test files; labels require separate semantic review.')
    additional_studies: list[str] = Field(default_factory=list,description='Additional completed study files, e.g. newly exported physical results.')
    correctness_supplements: list[str] = Field(default_factory=list,description='Source-bound interface audits or guarded affine rechecks used only for historical proofs that predate inline completeness fields.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(generator,params):
    _,op=find_op(generator)
    return op.In.model_validate(params).model_dump(exclude={'backend'})


def identity(generator,a,b,objective):
    return hashlib.sha256(json.dumps([generator,canonical(generator,a),canonical(generator,b),objective],sort_keys=True).encode()).hexdigest()


def studies(value):
    if isinstance(value,dict):
        if isinstance(value.get('data'),dict) and {'checkpoint_directory','cases'}<=value['data'].keys():
            yield value
        else:
            for child in value.values(): yield from studies(child)
    elif isinstance(value,list):
        for child in value: yield from studies(child)


def source_check(row,designs,supplements=None):
    """Bind correctness and measured wrappers to exact surviving source bytes."""
    physical=[];core=[]
    for design in designs:
        if 'timing_files' in design:
            files=[Path(x) for x in design['files']]
            core.append(fingerprint(files,design['top']))
            physical.append(fingerprint([Path(x) for x in design['timing_files']],design['timing_top']))
            if core[-1]!=design['source_sha256'] or physical[-1]!=design['timing_sha256']:
                raise InvalidInput('Generated sequential source identity changed')
        else:
            files=[Path(design['core_path'])]
            core.append(fingerprint(files,'dut'))
            physical.append(fingerprint(files+[Path(design['wrapper_path'])],'registered_dut'))
            if physical[-1]!=design['source_sha256']: raise InvalidInput('Generated combinational source identity changed')
    if 'checks' in row:
        if len(row['checks'])!=4: raise InvalidInput('Need all four source-bound core/fixture checks')
        for check,expected in zip(row['checks'],[core[0],physical[0],core[1],physical[1]]):
            data=check.get('data',{})
            if not check.get('ok') or not data.get('passed') or not data.get('source_unchanged') or data.get('source_sha256')!=expected:
                raise InvalidInput('Sequential correctness missing or mismatched')
    else:
        for name in ('simulation','formal'):
            check=row.get(name,{})
            flag='passed' if name=='simulation' else 'proved'
            if not check.get('ok') or not check.get('data',{}).get(flag) or [check.get('data',{}).get(k) for k in ('reference_sha256','candidate_sha256')]!=core:
                raise InvalidInput('Combinational correctness missing or mismatched')
        proof=row['formal'];data=proof['data'];complete=(data.get('interface_complete') is True and data.get('abstraction_defined') is True)
        supplements=supplements or {'interfaces':set(),'affine':set()}
        if not complete:
            if data.get('method')=='exact_affine_gf2':
                complete=tuple(core) in supplements['affine']
            else:
                complete=all(
                    (role,sha) in supplements['interfaces']
                    for role,sha in zip(('reference','candidate'),core))
        if not complete:
            raise InvalidInput('Formal proof lacks complete-interface and total-semantics evidence')
        cross=row.get('sat_crosscheck')
        if cross and not cross.get('ok') and not cross.get('data',{}).get('timed_out'):
            raise InvalidInput('SAT cross-check failed without a timeout')
    for pair in row['records']:
        if len(pair)!=2 or [r['source_sha256'] for r in pair]!=physical:
            raise InvalidInput('Measured sources differ from checked wrappers')
        for record,design in zip(pair,designs):
            ii=design['initiation_interval']
            latency=design.get('latency_cycles',design.get('minimum_latency_cycles',1))+design.get('fixture_observation_delay_cycles',0)
            if record['initiation_interval']!=ii or record['latency_cycles']!=latency:
                raise InvalidInput('Physical cycle contract differs from generated/checkable contract')
        if any(r['stage']!='routed_clocked_ooc' or r['tool']!='vivado' for r in pair):
            raise InvalidInput('Synthetic or nonphysical records cannot qualify')
    return physical


def correctness_supplements(paths,cwd,inputs):
    """Load narrowly recognized, source-bound historical proof supplements."""
    out={'interfaces':set(),'affine':set()}
    for name in paths:
        path=(Path(cwd)/name).resolve()
        if not path.is_file(): raise InvalidInput('Correctness supplement does not exist: '+name)
        obj=json.loads(path.read_text());inputs[str(path)]=digest(path)
        if (isinstance(obj,dict) and obj.get('passed') is True
                and str(obj.get('scope','')).startswith('Interface and total-binary source-netlist re-elaboration')
                and isinstance(obj.get('rows'),list)):
            if not obj['rows'] or any(row.get('interface_complete') is not True or row.get('abstraction_defined') is not True for row in obj['rows']):
                raise InvalidInput('Correctness supplement contains a failed source audit: '+name)
            for row in obj['rows']:
                out['interfaces'].add((row.get('role'),row.get('source_sha256')))
            continue
        if isinstance(obj,list):
            for row in obj:
                result=row.get('result',{});data=result.get('data',{});pair=tuple(row.get('source_sha256',[]))
                guards=data.get('source_guards',[])
                guards_ok=(len(guards)==2 and all(g.get('ok') and g.get('data',{}).get('proved')
                    and g.get('data',{}).get('interface_complete') and g.get('data',{}).get('abstraction_defined') for g in guards))
                if (len(pair)==2 and result.get('ok') and data.get('proved') and guards_ok
                        and tuple(data.get(k) for k in ('reference_sha256','candidate_sha256'))==pair):
                    out['affine'].add(pair)
            continue
        raise InvalidInput('Unrecognized correctness supplement format: '+name)
    return out


def timing_checks(row,designs,ctx):
    audited=0;historical=0
    for pair in row['records']:
        for record,design in zip(pair,designs):
            if not {'coverage','constraint_checks'} & record['evidence'].keys():
                historical+=1
                continue
            files=design.get('timing_files',[design.get('core_path'),design.get('wrapper_path')])
            top=design.get('timing_top','registered_dut')
            audit=Audit().timing_constraint_audit(AuditIn(record=record,files=files,top=top),ctx)
            if not audit.ok: raise InvalidInput('Timing coverage rejected: '+'; '.join(audit.data['reasons']))
            audited+=1
    return dict(audited_records=audited,historical_unaudited_records=historical)


@backend('synth','acceptance')
class Acceptance:
    def acceptance_audit(self,p,ctx):
        root=(Path(ctx.cwd)/p.experiment_root).resolve()
        evidence=(Path(ctx.cwd)/p.evidence_root).resolve()
        invpath=(Path(ctx.cwd)/p.inventory).resolve()
        if not root.is_dir() or not evidence.is_dir() or not invpath.is_file():
            raise InvalidInput('Experiment/evidence roots and inventory must exist')
        inputs={str(invpath):digest(invpath)}
        supplements=correctness_supplements(p.correctness_supplements,ctx.cwd,inputs)
        inventory=json.loads(invpath.read_text())
        tools=[];seen_names=set();seen_ids=set();bodies=set()
        for entry in inventory:
            errors=[]
            name=entry['operation'];number=entry['id']
            if name in seen_names or number in seen_ids: raise InvalidInput('Duplicate operation name or ID')
            seen_names.add(name);seen_ids.add(number)
            try:
                category,op=find_op(name)
                backend_name=op.In.model_fields['backend'].default or category.default_backend
                method=getattr(category.backends[backend_name].impl,name)
                implementation=Path(inspect.getsourcefile(method))
                module=ast.parse(implementation.read_text())
                owner=next(node for node in ast.walk(module) if isinstance(node,ast.ClassDef) and node.name==category.backends[backend_name].impl.__name__)
                function=next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name==name)
                function.name='operation'
                signature=ast.dump(function,include_attributes=False)
                if signature in bodies: errors.append('Duplicate implementation body')
                bodies.add(signature)
                implementation=Path(inspect.getsourcefile(method));inputs[str(implementation)]=digest(implementation)
            except (KeyError,AttributeError,TypeError,OSError,StopIteration) as exc: errors.append(str(exc))
            folders=list(root.glob(f'exp-tool-{number:02d}-*'))
            if len(folders)!=1: errors.append('Need one modular experiment directory')
            else:
                readme=folders[0]/'README.md'
                if not readme.is_file(): errors.append('Missing experiment README')
                else: inputs[str(readme)]=digest(readme)
            paper=entry.get('paper','');parsed=urlsplit(paper)
            if parsed.scheme!='https' or not parsed.netloc or paper[-1:] in ',.;': errors.append('Missing or malformed primary paper reference')
            if not entry.get('purpose'): errors.append('Missing independent engineering purpose')
            tests=entry.get('tests',[])
            if not tests: errors.append('Missing test references')
            for namepath in tests:
                path=(Path(ctx.cwd)/namepath).resolve()
                if not path.is_file(): errors.append('Missing test file: '+namepath)
                else: inputs[str(path)]=digest(path)
            tools.append(dict(id=number,operation=name,direct_ppa=entry.get('direct_ppa') is True,structurally_present=not errors,errors=errors))
        declarations={};unsupported=[]
        for path in sorted(root.glob('exp-tool-*/*.json')):
            obj=json.loads(path.read_text());inputs[str(path)]=digest(path)
            if not isinstance(obj,dict): continue
            cases=obj.get('cases',[])
            # Legacy multioperand study predates architecture_sweep, but remains
            # in the denominator rather than disappearing through format changes.
            if obj.get('op')=='synth_adder_tree' and 'widths' in obj:
                cases=[dict(generator='synth_adder_tree',baseline=dict(width=w,lanes=obj['lanes'],architecture='serial'),candidate=dict(width=w,lanes=obj['lanes'],architecture=a),objective='throughput') for w in obj['widths'] for a in obj['architectures'] if a!='serial']
            elif obj.get('op')=='synth_fifo' and 'configurations' in obj:
                cases=[dict(generator='synth_fifo',baseline=dict(**geometry,architecture='shift'),candidate=dict(**geometry,architecture='circular'),objective=obj['objective']) for geometry in obj['configurations']]
            elif not cases and ('widths' in obj or 'architectures' in obj) and obj.get('op') not in ('timing_constraint_audit','latency_throughput'):
                unsupported.append(str(path))
            for case in cases:
                key=identity(case['generator'],case['baseline'],case['candidate'],case['objective'])
                if key not in declarations:
                    declarations[key]=dict(generator=case['generator'],baseline=canonical(case['generator'],case['baseline']),candidate=canonical(case['generator'],case['candidate']),objective=case['objective'],declarations=[],observations=[])
                declarations[key]['declarations'].append(str(path))
        unmatched=[];seen_studies=set()
        paths=sorted(evidence.rglob('*.json'))+[(Path(ctx.cwd)/x).resolve() for x in p.additional_studies]
        for path in paths:
            obj=json.loads(path.read_text());inputs[str(path)]=digest(path)
            for study in studies(obj):
                serial=json.dumps(study,sort_keys=True)
                if serial in seen_studies: continue
                seen_studies.add(serial)
                directory=Path(study['data']['checkpoint_directory'])
                for row in study['data']['cases']:
                    observation=dict(file=str(path),run_id=study.get('run_id'),case=row['name'],status=row['status'],qualified=False)
                    try:
                        generated=[]
                        for role in ('baseline','candidate'):
                            checkpoint=directory/(row['name']+'-'+role+'-generate.json')
                            env=json.loads(checkpoint.read_text());inputs[str(checkpoint)]=digest(checkpoint)
                            if env['operation']!=row['generator'] or not env['result']['ok']: raise InvalidInput('Generation checkpoint mismatch')
                            generated.append(env)
                        key=identity(row['generator'],generated[0]['payload'],generated[1]['payload'],row['objective'])
                        if key not in declarations:
                            unmatched.append(dict(**observation,reason='Observed case absent from current declarations'));continue
                        target=declarations[key];target['observations'].append(observation)
                        if row['status']!='measured': continue
                        designs=[g['result']['data'] for g in generated]
                        source_check(row,designs,supplements)
                        observation['timing_coverage']=timing_checks(row,designs,ctx)
                        params=RepeatsIn(pairs=[ClockPair(baseline=a,candidate=b) for a,b in row['records']],objective=row['objective'])
                        summary=PhysicalAnalysis().paired_repeat_summary(params,ctx).data
                        observation.update(valid_physical=True,summary=summary,qualified=summary['area_gate'] or summary['throughput_gate'])
                    except (InvalidInput,ValueError,KeyError,OSError) as exc:
                        observation['error']=str(exc)
                        if not any(observation is x for d in declarations.values() for x in d['observations']): unmatched.append(observation)
        cases=[];ratios=[];qualifying={};measured={}
        for key,case in declarations.items():
            valid=[o for o in case['observations'] if o.get('valid_physical') and o['summary']['repeats']>=3]
            # Do not select the most favorable retry: retain the worst verified
            # complete result for this exact declared candidate and objective.
            ratio=min((o['summary']['objective_geometric_benefit_lower_bound'] for o in valid),default=None)
            geometry={k:v for k,v in case['baseline'].items() if k not in ('architecture','encoding','memory_style','implementation')}
            geometry_key=json.dumps(geometry,sort_keys=True)
            family=case['generator']
            substantial=any(isinstance(geometry.get(k),int) and geometry[k]>=limit for k,limit in [('width',16),('state_width',16),('states',64),('depth',64),('data_width',64)])
            if valid and substantial: measured.setdefault(family,set()).add(geometry_key)
            qualifies=bool(valid and substantial and all(o['qualified'] for o in valid))
            if qualifies: qualifying.setdefault(family,set()).add(geometry_key)
            if ratio is not None: ratios.append(ratio)
            cases.append(dict(id=key,**case,substantial=substantial,qualified=qualifies,objective_benefit_lower_bound=ratio))
        complete=bool(cases) and len(ratios)==len(cases) and not unmatched and not unsupported
        aggregate=math.exp(sum(math.log(r) for r in ratios)/len(ratios)) if complete else None
        inventory_ok=len(tools)==50 and seen_ids==set(range(1,51)) and all(t['structurally_present'] for t in tools)
        category_families={
            'arithmetic':{'synth_mcm','synth_serial_multiplier','synth_divider','synth_prefix_adder','synth_fir','synth_dot_product','synth_systolic_tile','synth_saturating_alu'},
            'reduction':{'synth_adder_tree','synth_popcount','synth_argmax_tree'},
            'selection_control':{'synth_priority_encoder','synth_leading_zero','synth_onehot_mux','synth_fsm'},
            'streaming_sequential':{'synth_fifo','synth_skid_buffer','synth_systolic_tile'},
            'memory':{'synth_fifo','synth_banked_regfile'}}
        category_coverage={k:any(len(measured.get(f,set()))>=2 for f in families) for k,families in category_families.items()}
        gates=dict(inventory_structure=inventory_ok,category_coverage=all(category_coverage.values()),direct_ppa_labels=sum(t['direct_ppa'] for t in tools)>=40,
                   diverse_measured_families=sum(len(v)>=2 for v in measured.values())>=12,
                   qualifying_families=sum(len(v)>=2 for v in qualifying.values())>=6,
                   all_declared_cases_measured=complete,all_case_objective_benefit=aggregate is not None and aggregate>=1.10)
        data=dict(gates=gates,automated_gates_passed=all(gates.values()),tools=tools,cases=cases,
                  category_coverage=category_coverage,declared_cases=len(cases),cases_with_verified_physical_pairs=len(ratios),
                  qualifying_family_configurations={k:len(v) for k,v in qualifying.items()},
                  measured_family_configurations={k:len(v) for k,v in measured.items()},
                  all_case_geometric_benefit_lower_bound=aggregate,unmatched_observations=unmatched,
                  unsupported_declarations=unsupported,input_sha256=inputs,
                  remaining_review=['Substantive distinct operation purposes and direct PPA classification','Meaningful negative tests and actual per-operation experiment coverage','Historical timing coverage and report authenticity','Declaration timing, objective preservation and full historical inventory','Required documents, OpenSpec, PR state and usage stop rule'],
                  completion_claim_supported=False)
        (ctx.run.artifacts/'acceptance.json').write_text(json.dumps(data,indent=2)+'\n')
        return Result(ok=all(gates.values()),data=data,note='This is a bounded mechanical acceptance audit, not a completion certificate. Missing, failed or unmatched cases keep the full aggregate undefined; no neutral ratio is imputed. Every complete retry is retained and the worst benefit is used. Structural inventory does not prove semantic distinctness or test quality. Remaining review items require independent evidence before whole-goal completion.')


CATEGORY.ops.append(Op('acceptance_audit',AcceptanceIn,Result,'Audit complete declarations, source-bound correctness, physical repeats, diversity and all-case PPA gates.'))
