"""Actual SAT controls for the complete, exactly declared packed interface."""
import argparse
import json
import os
from pathlib import Path
from ic_core import dispatch

here = Path(__file__).resolve().parent
os.chdir(here.parents[2])
parser = argparse.ArgumentParser()
parser.add_argument('--container')
args = parser.parse_args()
out = here / 'output' / 'interfaces'
out.mkdir(parents=True, exist_ok=True)
plain = 'module dut(input [7:0] x,output [7:0] y); assign y=x; endmodule'
extra = lambda value: ('module dut(input [7:0] x,output [7:0] y,output z); '
                       f"assign y=x; assign z=1'b{value}; endmodule")
cases = [
    ('complete', plain, plain, True),
    ('hidden_output_mismatch', extra(0), extra(1), False),
    ('reference_extra_output', extra(0), plain, False),
    ('candidate_extra_output', plain, extra(0), False),
    ('unused_extra_input', plain, plain.replace('output [7:0] y', 'input hidden,output [7:0] y'), False),
    ('inout_y', plain, plain.replace('output [7:0] y', 'inout [7:0] y'), False),
    ('wrong_input_width', plain, plain.replace('input [7:0] x', 'input [8:0] x'), False),
    ('wrong_output_width', plain, plain.replace('output [7:0] y', 'output [8:0] y'), False),
]
rows = []
for mode in ('word', 'aig', 'macc', 'bitwise', 'bitwise_products', 'bitwise_prefix'):
    for label, reference, candidate, expected in cases:
        ref, cand = out / 'reference.sv', out / 'candidate.sv'
        ref.write_text(reference)
        cand.write_text(candidate)
        result = dispatch('yosys_equivalence', dict(
            reference_files=[str(ref)], candidate_files=[str(cand)], top='dut',
            input_width=8, output_width=8, normalization=mode,
            container=args.container, timeout_s=30))
        rows.append(dict(mode=mode, label=label, expected=expected, result=result))
        (out / 'controls.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(mode, label, result['ok'], result['run_id'], flush=True)
        assert result['ok'] == expected
        assert result['data']['interface_complete'] == expected
