"""Optimization advice from a small transparent, paper-attributed rule library."""
from typing import Literal
from pydantic import Field, model_validator
from ...registry import Op, backend
from ..exploration_common import ExplorationInput
from ..synth.exploration import Result
from . import CATEGORY

RULES = [
    {'id': 'share-adder', 'topics': ['area', 'mux'], 'rewrite': 'sel ? (a+b) : (a+c) -> a + (sel ? b : c)',
     'preconditions': ['Identical modular widths and signedness', 'Combinational outputs', 'Check mux-before-adder timing trade-off'], 'paper': 'RTLRewriter', 'url': 'https://arxiv.org/abs/2409.11414'},
    {'id': 'factor-product', 'topics': ['area', 'arithmetic'], 'rewrite': 'a*b + a*c -> a*(b+c)',
     'preconditions': ['Preserve all intermediate truncation semantics', 'Prove modular equivalence', 'Re-measure DSP and timing trade-offs'], 'paper': 'ASPEN', 'url': 'https://www.csl.cornell.edu/~zhiruz/pdfs/aspen-mlcad2025.pdf'},
    {'id': 'constant-multiply', 'topics': ['area', 'arithmetic'], 'rewrite': 'x*constant -> shift/add candidate',
     'preconditions': ['Explicit signed extension and result width', 'Compare against native DSP inference', 'Avoid assuming fewer RTL operators means less area'], 'paper': 'ASPEN', 'url': 'https://www.csl.cornell.edu/~zhiruz/pdfs/aspen-mlcad2025.pdf'},
    {'id': 'fsm-merge', 'topics': ['area', 'fsm'], 'rewrite': 'Merge states with matching output and transition behavior',
     'preconditions': ['Prove sequential equivalence including reset', 'Preserve unspecified-state semantics', 'Not supported by the combinational checker'], 'paper': 'SymRTLO', 'url': 'https://arxiv.org/abs/2504.10369'},
    {'id': 'pipeline', 'topics': ['delay', 'pipeline'], 'rewrite': 'Insert a register boundary on a critical datapath',
     'preconditions': ['Explicitly allow latency changes', 'Update protocol and testbench', 'Account for register area and power'], 'paper': 'Mascot', 'url': 'https://research.ibm.com/publications/enhancing-llms-for-hdl-code-optimization-using-domain-knowledge-injection'},
    {'id': 'clock-enable', 'topics': ['power', 'enable'], 'rewrite': 'Use an enable when a register value need not change',
     'preconditions': ['Use FPGA clock enables, never an arbitrary gated clock', 'Prove sequential behavior', 'Power needs representative switching activity'], 'paper': 'Mascot', 'url': 'https://research.ibm.com/publications/enhancing-llms-for-hdl-code-optimization-using-domain-knowledge-injection'},
]


class RulesIn(ExplorationInput):
    backend: str | None = Field(default='exploration', description='Paper-informed advice backend.')
    topics: list[str] = Field(default=['area'], description='Topic keywords; matches explicit rule tags, case-insensitively.')


class WidthIn(ExplorationInput):
    backend: str | None = Field(default='exploration', description='Paper-informed advice backend.')
    a_min: int = Field(description='Inclusive lower bound for operand a.')
    a_max: int = Field(description='Inclusive upper bound for operand a.')
    b_min: int = Field(description='Inclusive lower bound for operand b.')
    b_max: int = Field(description='Inclusive upper bound for operand b.')
    operation: Literal['add', 'sub', 'mul'] = Field(description='Integer arithmetic operation to bound before RTL truncation.')
    declared_width: int = Field(ge=1, le=65536, description='Current result width to assess.')

    @model_validator(mode='after')
    def valid_ranges(self):
        if self.a_min > self.a_max or self.b_min > self.b_max:
            raise ValueError('range lower bound exceeds upper bound')
        if any(abs(v).bit_length() > 65536 for v in (self.a_min, self.a_max, self.b_min, self.b_max)):
            raise ValueError('operand range exceeds supported width')
        return self


@backend('lint', 'exploration')
class Advice:
    def optimization_rules(self, params, ctx):
        topics = {t.lower() for t in params.topics}
        return Result(data={'rules': [r for r in RULES if topics.intersection(r['topics'])]},
                      note='Curated retrieval inspired by RTLRewriter/SymRTLO/Mascot/ASPEN. Advice only; no automatic rewrite or correctness claim.')

    def width_advice(self, params, ctx):
        if params.operation == 'add':
            low, high = params.a_min + params.b_min, params.a_max + params.b_max
        elif params.operation == 'sub':
            low, high = params.a_min - params.b_max, params.a_max - params.b_min
        else:
            products = [a*b for a in (params.a_min, params.a_max) for b in (params.b_min, params.b_max)]
            low, high = min(products), max(products)
        signed = low < 0
        width = max(1, high.bit_length()) if not signed else max(1, (~low).bit_length() + 1, max(0, high).bit_length() + 1)
        return Result(data={'min': low, 'max': high, 'signed': signed, 'required_width': width,
                            'potential_bit_reduction': max(0, params.declared_width - width),
                            'overflow_possible': width > params.declared_width},
                      note='Exact interval bound for independent mathematical integers. User ranges must be established; preserve RTL signedness, overflow, and truncation. Width reduction is not a measured PPA gain.')


CATEGORY.ops.extend([
    Op('optimization_rules', RulesIn, Result, 'Retrieve attributed PPA rewrite rules with correctness preconditions.'),
    Op('width_advice', WidthIn, Result, 'Bound arithmetic ranges to find potential result-width reduction.'),
])
