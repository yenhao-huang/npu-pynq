"""Generate modular reduction alternatives with shared registered boundaries."""
from typing import Literal
from pydantic import Field
from ...registry import Op, backend
from ..exploration_common import ExplorationInput
from . import CATEGORY
from .exploration import Result


class ReductionIn(ExplorationInput):
    backend: str | None = Field(default='reduction_rtl', description='Operation-specific execution backend.')
    width: int = Field(default=32, ge=8, le=128, description='Unsigned operand and modular result width.')
    lanes: int = Field(default=16, ge=3, le=128, description='Number of reduction operands.')
    architecture: Literal['serial', 'balanced', 'compressor'] = Field(default='balanced', description='Reduction implementation to generate.')


def reduction_rtl(width, lanes, architecture):
    lines = [f'module dut(input [{width*lanes-1}:0] x, output [{width-1}:0] y);']
    terms = [f'x[{i*width} +: {width}]' for i in range(lanes)]
    counter = 0
    def wire(expression):
        nonlocal counter
        name = f'n{counter}'
        counter += 1
        lines.append(f'wire [{width-1}:0] {name} = {expression};')
        return name
    if architecture == 'serial':
        total = terms[0]
        for term in terms[1:]:
            total = wire(f'{total} + {term}')
    else:
        while len(terms)>2:
            next_terms=[]
            step=3 if architecture=='compressor' else 2
            for i in range(0,len(terms),step):
                group=terms[i:i+step]
                if len(group)<step:
                    next_terms.extend(group)
                elif step==2:
                    next_terms.append(wire(f'{group[0]} + {group[1]}'))
                else:
                    a,b,c=group
                    next_terms.append(wire(f'{a} ^ {b} ^ {c}'))
                    next_terms.append(wire(f'(({a} & {b}) | ({a} & {c}) | ({b} & {c})) << 1'))
            terms=next_terms
        total=wire(' + '.join(terms))
    lines += [f'assign y={total};','endmodule',
              f'module registered_dut(input clk, input [{width*lanes-1}:0] x, output reg [{width-1}:0] y);',
              f'reg [{width*lanes-1}:0] x_q;', f'wire [{width-1}:0] result;',
              'dut core(.x(x_q), .y(result));',
              'always @(posedge clk) begin x_q <= x; y <= result; end', 'endmodule']
    return '\n'.join(lines)+'\n'


@backend('synth', 'reduction_rtl')
class ReductionGenerator:
    def synth_adder_tree(self, params, ctx):
        # Combinational and registered sources are split so conservative checking
        # never silently accepts the sequential wrapper.
        text = reduction_rtl(params.width, params.lanes, params.architecture)
        core, wrapper = text.split('module registered_dut',1)
        path=ctx.run.artifacts/'core.sv'
        path.write_text(core,encoding='utf-8')
        registered=ctx.run.artifacts/'registered.sv'
        registered.write_text('module registered_dut'+wrapper,encoding='utf-8')
        return Result(data=dict(core=ctx.run.handle(path.name), wrapper=ctx.run.handle(registered.name),
                               core_path=str(ctx.run.final_artifact(path.name)), wrapper_path=str(ctx.run.final_artifact(registered.name)), input_width=params.width*params.lanes,
                               output_width=params.width, latency_cycles=1, initiation_interval=1,
                               architecture=params.architecture), note='Unsigned modular sum. Serial, balanced carry-propagate, or carry-save reduction; registered wrapper is identical across alternatives. PPA benefit requires measurement.')


CATEGORY.ops.append(Op('synth_adder_tree', ReductionIn, Result, 'Generate parameterized modular reduction RTL with serial, balanced, or carry-save architecture.'))
