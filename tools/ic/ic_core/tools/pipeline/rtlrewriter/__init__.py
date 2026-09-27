"""Verification-gated PPA evaluation adapted from RTLRewriter."""
from pydantic import Field
from ....registry import Op, backend
from ....dispatch import dispatch
from ...debug.comb import CheckIn
from ...synth.exploration import Result
from ...synth.measure import MeasureIn
from .. import CATEGORY


class EvaluateIn(CheckIn):
    backend: str | None = Field(default='rtlrewriter', description='RTLRewriter-inspired verification/measurement composition.')
    device: str = Field(default='xc7z020clg400-1', pattern=r'^[A-Za-z0-9_-]+$', description='Vivado device for both designs.')
    constraint_ns: float = Field(default=10, gt=0, allow_inf_nan=False, description='Common combinational input-to-output constraint in ns.')
    measure_timeout_s: float = Field(default=300, gt=0, le=3600, allow_inf_nan=False, description='Maximum runtime for each Vivado measurement.')


@backend('pipeline', 'rtlrewriter')
class Evaluate:
    def rtl_evaluate(self, params, ctx):
        children = []
        check_payload = {k: v for k, v in params.model_dump().items() if k in CheckIn.model_fields and k != 'backend'}
        check = dispatch('comb_check', check_payload, cwd=ctx.cwd, store=ctx.store)
        children.append(check['run_id'])
        if not check['ok']:
            return Result(ok=False, data={'stage': 'correctness', 'check': check, 'children': children}, note='PPA runs skipped because the combinational check failed.')
        records = []
        for name, files in [('baseline', params.reference_files), ('candidate', params.candidate_files)]:
            result = dispatch('ppa_measure', {'name': name, 'files': files, 'top': params.top, 'device': params.device,
                                             'constraint_ns': params.constraint_ns, 'timeout_s': params.measure_timeout_s}, cwd=ctx.cwd, store=ctx.store)
            children.append(result['run_id'])
            if not result['ok']:
                return Result(ok=False, data={'stage': name + '_measurement', 'result': result, 'children': children}, note='Measurement failed; no optimization claim.')
            record = result['data']['record']
            expected = check['data']['reference_sha256' if name == 'baseline' else 'candidate_sha256']
            if record['source_sha256'] != expected:
                return Result(ok=False, data={'stage': 'source_changed', 'children': children}, note='Source changed between correctness checking and measurement.')
            records.append(record)
        comparison = dispatch('ppa_compare', {'baseline': records[0], 'candidate': records[1]}, cwd=ctx.cwd, store=ctx.store)
        children.append(comparison['run_id'])
        return Result(data={'check': check, 'records': records, 'comparison': comparison, 'children': children},
                      note='Correctness-gated measured comparison. Full RTLRewriter search, LLM generation, and formal proof are not implemented.')


CATEGORY.ops.append(Op('rtl_evaluate', EvaluateIn, Result, 'Check a combinational rewrite, measure both designs, and compare compatible PPA.', long_running=True))
from . import sweep  # noqa: E402,F401
