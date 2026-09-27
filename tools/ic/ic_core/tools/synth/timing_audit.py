"""Fail-closed consistency and constraint-coverage audit for routed records."""
import hashlib
import math
import re
from pydantic import Field
from ...errors import InvalidInput
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, Identifier, fingerprint, sources
from . import CATEGORY
from .exploration import Result
from .physical_analysis import ClockRecord

CHECKS=('no_clock','constant_clock','multiple_clock','loops','latch_loops','unconstrained_internal_endpoints')


def parse_checks(text):
    counts={}
    for name in CHECKS:
        values=re.findall(r'^\s*\d+\.\s+checking\s+'+name+r'\s+\((\d+)\)\s*$',text,re.M|re.I)
        if len(values)!=2 or len(set(values))!=1:
            raise InvalidInput('Missing, duplicated or inconsistent timing check: '+name)
        counts[name]=int(values[0])
        section=re.split(r'^\s*\d+\.\s+checking\s+'+name+r'\s+\(\d+\)\s*$',text,flags=re.M|re.I)[-1]
        section=re.split(r'^\s*\d+\.\s+checking\s+',section,flags=re.M|re.I)[0]
        details=re.findall(r'There (?:are|is) (\d+) ',section)
        if not details or (counts[name]==0 and any(int(value) for value in details)):
            raise InvalidInput('Inconsistent or missing timing-check details: '+name)
    return counts


def fields(text):
    out={}
    for line in text.splitlines():
        if '=' not in line: continue
        key,value=line.split('=',1)
        if key in out: raise InvalidInput('Duplicate audit field: '+key)
        out[key]=value
    return out


class AuditIn(ExplorationInput):
    backend: str | None = Field(default='timing_audit',description='Source-bound physical record audit backend.')
    record: ClockRecord = Field(description='Routed clocked record with coverage and check_timing report handles from the same run.')
    files: list[str] = Field(min_length=1,description='Exact measured RTL sources, including the registered fixture.')
    top: Identifier = Field(description='Measured top module used for source identity and report binding.')


@backend('synth','timing_audit')
class Audit:
    def timing_constraint_audit(self,p,ctx):
        r=p.record
        if r.stage!='routed_clocked_ooc' or r.tool!='vivado': raise InvalidInput('Audit requires a real routed Vivado record')
        required=('metrics','timing','coverage','constraint_checks')
        if any(name not in r.evidence for name in required): raise InvalidInput('Missing constraint coverage evidence; historical record is unaudited')
        handles=[r.evidence[name] for name in required]
        if any(not re.fullmatch(r'[0-9a-f]{6}/[A-Za-z0-9_.-]+',handle) for handle in handles) or len({handle.split('/')[0] for handle in handles})!=1:
            raise InvalidInput('Timing evidence must belong to the same run')
        files=sources(p.files,ctx.cwd);sha=fingerprint(files,p.top)
        if sha!=r.source_sha256: raise InvalidInput('Measured source identity mismatch')
        paths={name:ctx.resolve(r.evidence[name]) for name in required}
        contents={name:path.read_text(encoding='utf-8') for name,path in paths.items()}
        digests={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
        coverage=fields(contents['coverage']);metrics=fields(contents['metrics'])
        if coverage.get('source_sha256')!=sha or coverage.get('top')!=p.top:
            raise InvalidInput('Coverage source/top identity mismatch')
        if coverage.get('scope')!='single_clock_register_to_register_ooc' or coverage.get('io_delays_constrained')!='0':
            raise InvalidInput('Unsupported timing scope')
        if metrics.get('version')!=r.version or metrics.get('build')!=r.build:
            raise InvalidInput('Timing tool/build mismatch')
        if not re.search(r'^\|\s*Design\s*:\s*'+re.escape(p.top)+r'\s*$',contents['constraint_checks'],re.M):
            raise InvalidInput('Timing-check design identity mismatch')
        try:
            integers={name:int(coverage[name]) for name in ('clock_count','register_count','clocked_register_count','latch_count','setup_path_count')}
            if any(value<0 for value in integers.values()): raise ValueError('negative count')
            period=float(coverage['clock_period_ns']);slack=float(metrics['slack_ns']);requirement=float(metrics['requirement_ns'])
            if not all(math.isfinite(v) for v in (period,slack,requirement)): raise ValueError('nonfinite timing')
        except (ValueError,KeyError) as exc: raise InvalidInput('Malformed timing coverage values') from exc
        if not math.isclose(period,r.period_ns,abs_tol=1e-6) or not math.isclose(requirement,r.period_ns,abs_tol=.01) or not math.isclose(slack,r.metrics['slack_ns'],abs_tol=1e-6):
            raise InvalidInput('Clock or timing metrics disagree with physical record')
        timing=contents['timing']
        report_slack=re.findall(r'^Slack \((?:MET|VIOLATED)\)\s*:\s*([-+0-9.]+)ns',timing,re.M)
        if (not re.search(r'^\|\s*Design State\s*:\s*Routed\s*$',timing,re.M)
            or not re.search(r'^\|\s*Design\s*:\s*'+re.escape(p.top)+r'\s*$',timing,re.M)
            or not re.search(r'Path Type:\s+Setup',timing)
            or not re.search(r'Path Group:\s+ic_clock\s*$',timing,re.M)
            or len(report_slack)!=1 or not math.isclose(float(report_slack[0]),slack,abs_tol=.0011)):
            raise InvalidInput('Routed setup report disagrees with the audited record')
        counts=parse_checks(contents['constraint_checks'])
        reasons=[name+' has '+str(count)+' violations' for name,count in counts.items() if count]
        if integers['clock_count']!=1: reasons.append('Expected exactly one clock')
        if not integers['register_count'] or integers['clocked_register_count']!=integers['register_count']:
            reasons.append('Some sequential cells are outside the measured clock')
        if integers['latch_count']: reasons.append('Level-sensitive storage is outside this audit contract')
        if integers['setup_path_count']!=1: reasons.append('Missing register-to-register setup path')
        if sha!=fingerprint(files,p.top) or digests!={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}:
            raise InvalidInput('Evidence changed during audit')
        passed=not reasons
        return Result(ok=passed,data=dict(passed=passed,source_sha256=sha,scope=coverage['scope'],coverage=integers,violations=counts,reasons=reasons,
            evidence=dict(r.evidence),report_sha256=digests,period_ns=period,setup_slack_ns=slack,requested_setup_met=slack>=0,
            excluded=['top-level input/output delays','hold-time closure','board constraints','protocol correctness']),
            note='Audits one routed single-clock register-to-register setup scope. Zero coverage violations is not board timing closure; a negative setup slack remains a failed requested period. Original PPA records are not modified. Historical records without coverage reports remain unaudited.')


CATEGORY.ops.append(Op('timing_constraint_audit',AuditIn,Result,'Audit clock coverage and source/report consistency for routed setup measurements.'))
