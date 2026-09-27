"""Bounded exhaustive binary testing of explicitly combinational RTL."""
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ...process import run as run_process
from ...registry import Op, backend
from ...errors import InvalidInput
from ..exploration_common import ExplorationInput
from ..exploration_common import Identifier, fingerprint, sources, require_combinational
from ..synth.exploration import Result
from . import CATEGORY


class Port(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: Identifier
    width: int = Field(ge=1, le=256)


class CheckIn(ExplorationInput):
    backend: str | None = Field(default='comb_icarus', description='Icarus exhaustive combinational test backend.')
    reference_files: list[str] = Field(min_length=1, description='Reference RTL sources with no external includes.')
    candidate_files: list[str] = Field(min_length=1, description='Candidate RTL sources with no external includes.')
    top: Identifier = Field(description='Same top module name in both separately compiled designs.')
    inputs: list[Port] = Field(min_length=1, description='Complete combinational input interface in enumeration order, at most 16 bits total.')
    outputs: list[Port] = Field(min_length=1, description='Complete combinational output interface to compare.')
    combinational_contract: bool = Field(description='Caller confirms no state, latches, clock, reset, or timed behavior and a complete interface.')
    timeout_s: float = Field(default=30, gt=0, le=300, allow_inf_nan=False, description='Timeout in seconds for each compile or simulation process.')

    @model_validator(mode='after')
    def bounded(self):
        if not self.combinational_contract:
            raise ValueError('explicit combinational contract is required')
        names = [p.name for p in self.inputs + self.outputs]
        if len(names) != len(set(names)):
            raise ValueError('port names must be unique')
        if sum(p.width for p in self.inputs) > 16:
            raise ValueError('exhaustive checking is limited to 16 input bits')
        if sum(p.width for p in self.outputs) > 1024:
            raise ValueError('output interface exceeds 1024 bits')
        return self


@backend('debug', 'comb_icarus', requires='iverilog', version_cmd=['iverilog', '-V'])
class Checker:
    def comb_check(self, params, ctx):
        reference = sources(params.reference_files, ctx.cwd)
        candidate = sources(params.candidate_files, ctx.cwd)
        for files in (reference, candidate):
            require_combinational(files)
            if any('`include' in p.read_text(encoding='utf-8') for p in files):
                raise InvalidInput('inline include dependencies before exhaustive checking')
        reference_hash = fingerprint(reference, params.top)
        candidate_hash = fingerprint(candidate, params.top)
        count = 1 << sum(p.width for p in params.inputs)
        declarations = [f'reg [{p.width-1}:0] {p.name};' for p in params.inputs]
        declarations += [f'wire [{p.width-1}:0] {p.name};' for p in params.outputs]
        connections = ', '.join(f'.{p.name}({p.name})' for p in params.inputs + params.outputs)
        input_concat = '{' + ','.join(p.name for p in params.inputs) + '}'
        output_concat = '{' + ','.join(p.name for p in params.outputs) + '}'
        bench = ctx.run.work / 'tb.sv'
        bench.write_text('module ic_exploration_tb;\n' + '\n'.join(declarations) + f'''
{params.top} dut({connections});
integer i;
initial begin
  for (i = 0; i < {count}; i = i + 1) begin
    {input_concat} = i;
    #1;
    $display("IC_VECTOR %0d %h", i, {output_concat});
  end
  $finish;
end
endmodule
''', encoding='utf-8')
        tables = []
        handles = []
        for label, files in [('reference', reference), ('candidate', candidate)]:
            exe = ctx.run.work / (label + '.vvp')
            compile_log = ctx.run.artifacts / (label + '_compile.log')
            build = run_process(['iverilog', '-g2012', '-s', 'ic_exploration_tb', '-o', str(exe), *map(str, files), str(bench)],
                                cwd=ctx.run.work, log_path=compile_log, timeout_s=params.timeout_s)
            if build.exit_code or build.timed_out:
                return Result(ok=False, data={'stage': label + '_compile', 'log': ctx.run.handle(compile_log.name)}, note='Compilation failed or timed out; no correctness verdict.')
            log = ctx.run.artifacts / (label + '_vectors.log')
            run = run_process(['vvp', str(exe)], cwd=ctx.run.work, log_path=log, timeout_s=params.timeout_s)
            handles.append(ctx.run.handle(log.name))
            # This bounded tool owns and parses its transcript; clients receive handles.
            text = run.text(limit_bytes=24_000_000)
            rows = re.findall(r'^IC_VECTOR ([0-9]+) ([0-9a-fAxXzZ]+)\s*$', text, re.M)
            if (run.exit_code or run.timed_out or len(rows) != count or
                    [int(i) for i, _ in rows] != list(range(count)) or re.search(r'\b(ERROR|FATAL)\b', text)):
                return Result(ok=False, data={'stage': label + '_simulation', 'log': handles[-1]}, note='Simulation failed, timed out, or did not enumerate every input.')
            tables.append([value.lower() for _, value in rows])
        mismatch = next((i for i, (a, b) in enumerate(zip(*tables)) if a != b or any(c in a+b for c in 'xz')), None)
        if fingerprint(reference, params.top) != reference_hash or fingerprint(candidate, params.top) != candidate_hash:
            return Result(ok=False, data={}, note='Sources changed during checking; result discarded.')
        data = {'passed': mismatch is None, 'vectors': count, 'first_mismatch': mismatch,
                'reference_sha256': reference_hash, 'candidate_sha256': candidate_hash,
                'logs': handles}
        if mismatch is not None:
            data.update(reference_value=tables[0][mismatch], candidate_value=tables[1][mismatch])
        return Result(ok=mismatch is None, data=data, note='Exhaustive binary input simulation under the declared combinational contract. Not a sequential/formal equivalence proof; no X-input coverage.')


CATEGORY.ops.append(Op('comb_check', CheckIn, Result, 'Exhaustively test up to 16 binary input bits for combinational rewrite mismatches.', long_running=True))
