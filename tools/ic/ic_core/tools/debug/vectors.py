"""Reproducible directed and random checks for wide combinational interfaces."""
import hashlib
import random
import re

from pydantic import Field, model_validator
from ...errors import InvalidInput
from ...process import run as run_process
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, Identifier, fingerprint, sources, require_combinational
from ..synth.exploration import Result
from . import CATEGORY


class VectorIn(ExplorationInput):
    backend: str | None = Field(default='vector_icarus', description='Operation-specific execution backend.')
    reference_files: list[str] = Field(min_length=1, description='Reference self-contained SystemVerilog sources.')
    candidate_files: list[str] = Field(min_length=1, description='Candidate self-contained SystemVerilog sources.')
    top: Identifier = Field(description='Top module identifier, identical in separately compiled variants.')
    input_width: int = Field(ge=1, le=4096, description='Packed input port x width.')
    output_width: int = Field(ge=1, le=4096, description='Packed output port y width.')
    random_vectors: int = Field(default=4096, ge=1, le=100000, description='Number of seeded random vectors in addition to directed patterns.')
    seed: int = Field(default=928, ge=0, le=2**32-1, description='Deterministic random stimulus seed.')
    combinational_contract: bool = Field(description='Confirm the complete x/y interface has no state, latches, or timed behavior.')
    timeout_s: float = Field(default=60, gt=0, le=600, allow_inf_nan=False, description='Maximum process runtime in seconds.')

    @model_validator(mode='after')
    def contract(self):
        if not self.combinational_contract:
            raise ValueError('explicit complete combinational x/y interface contract required')
        return self


def stimuli(width, count, seed):
    mask = (1 << width) - 1
    alternating = sum(1 << i for i in range(0, width, 2))
    values = [0, mask, alternating, mask ^ alternating]
    for i in range(width):
        values.extend((1 << i, mask ^ (1 << i), (1 << i) - 1))
    rng = random.Random(seed)
    values.extend(rng.getrandbits(width) for _ in range(count))
    return values


@backend('debug', 'vector_icarus', requires='iverilog', version_cmd=['iverilog', '-V'])
class VectorChecker:
    def vector_equivalence(self, params, ctx):
        designs = [sources(params.reference_files, ctx.cwd), sources(params.candidate_files, ctx.cwd)]
        for files in designs:
            require_combinational(files)
            if any('`include' in p.read_text(encoding='utf-8') for p in files):
                raise InvalidInput('inline include dependencies before vector checking')
        hashes = [fingerprint(files, params.top) for files in designs]
        vectors = stimuli(params.input_width, params.random_vectors, params.seed)
        payload = ''.join(f'{v:x}\n' for v in vectors)
        (ctx.run.work / 'vectors.hex').write_text(payload, encoding='ascii')
        bench = ctx.run.work / 'tb.sv'
        bench.write_text(f'''module ic_vector_tb;
reg [{params.input_width-1}:0] x;
wire [{params.output_width-1}:0] y;
reg [{params.input_width-1}:0] vectors [0:{len(vectors)-1}];
{params.top} dut(.x(x), .y(y));
integer i;
initial begin
  $readmemh("vectors.hex", vectors);
  for (i=0; i<{len(vectors)}; i=i+1) begin
    x=vectors[i]; #1;
    $display("IC_VECTOR %0d %h", i, y);
  end
  $finish;
end
endmodule
''', encoding='utf-8')
        tables, handles = [], []
        for label, files in zip(('reference', 'candidate'), designs):
            exe = ctx.run.work / (label + '.vvp')
            name = label + '_compile.log'
            build = run_process(['iverilog', '-g2012', '-s', 'ic_vector_tb', '-o', str(exe), *map(str, files), str(bench)], cwd=ctx.run.work, log_path=ctx.run.artifacts/name, timeout_s=params.timeout_s)
            if build.exit_code or build.timed_out or re.search(r'warning:.*(port|dangling|width)', build.text(), re.I):
                return Result(ok=False, data={'log': ctx.run.handle(name),'exit_code':build.exit_code,'timed_out':build.timed_out}, note='Compilation failed or timed out; no verdict.')
            name = label + '_vectors.log'
            result = run_process(['vvp', str(exe)], cwd=ctx.run.work, log_path=ctx.run.artifacts/name, timeout_s=params.timeout_s)
            handles.append(ctx.run.handle(name))
            text = result.text(limit_bytes=(len(vectors)*(params.output_width//4+100)))
            rows = re.findall(r'^IC_VECTOR ([0-9]+) ([0-9a-fAxXzZ]+)\s*$', text, re.M)
            if (result.exit_code or result.timed_out or len(rows)!=len(vectors)
                    or [int(i) for i, _ in rows]!=list(range(len(vectors)))
                    or re.search(r'\b(ERROR|FATAL)\b', text)):
                return Result(ok=False, data={'log': handles[-1],'exit_code':result.exit_code,'timed_out':result.timed_out,'expected_vectors':len(vectors),'observed_vectors':len(rows)}, note='Incomplete or failed simulation; no verdict.')
            tables.append([value.lower() for _, value in rows])
        if hashes != [fingerprint(files, params.top) for files in designs]:
            return Result(ok=False, data={}, note='Sources changed; result discarded.')
        mismatch = next((i for i, (a,b) in enumerate(zip(*tables)) if a!=b or any(c in a+b for c in 'xz')), None)
        data = dict(passed=mismatch is None, vectors=len(vectors), seed=params.seed,
                    stimuli_sha256=hashlib.sha256(payload.encode('ascii')).hexdigest(),
                    reference_sha256=hashes[0], candidate_sha256=hashes[1], logs=handles,
                    first_mismatch=mismatch)
        if mismatch is not None:
            data.update(input_hex=f'{vectors[mismatch]:x}', reference_value=tables[0][mismatch], candidate_value=tables[1][mismatch])
        return Result(ok=mismatch is None, data=data, note='Directed and seeded binary simulation under caller-declared complete combinational interface; not formal or exhaustive equivalence.')


CATEGORY.ops.append(Op('vector_equivalence', VectorIn, Result, 'Check wide combinational rewrites with directed carry/walking patterns and reproducible random stimuli.', long_running=True))
