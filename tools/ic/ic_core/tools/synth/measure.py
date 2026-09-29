"""Local Vivado routed combinational measurement with explicit constraints."""
import shutil
from pydantic import Field

from ...process import run as run_process
from ...registry import Op, backend
from ...errors import InvalidInput
from ..exploration_common import ExplorationInput
from ..exploration_common import Identifier, fingerprint, sources, tcl_path, require_combinational
from . import CATEGORY, DEFAULT_PART
from .exploration import Record, Result


class MeasureIn(ExplorationInput):
    backend: str | None = Field(default='ppa_vivado', description='Vivado combinational measurement backend.')
    files: list[str] = Field(min_length=1, description='Self-contained SystemVerilog sources; no external includes.')
    top: Identifier = Field(description='Combinational top module.')
    name: str = Field(min_length=1, description='Candidate identifier.')
    device: str = Field(default=DEFAULT_PART, pattern=r'^[A-Za-z0-9_-]+$', description='Vivado FPGA device part.')
    constraint_ns: float = Field(default=10.0, gt=0, allow_inf_nan=False, description='Maximum input-to-output datapath delay constraint, ns.')
    timeout_s: float = Field(default=300, gt=0, le=3600, allow_inf_nan=False, description='Maximum Vivado runtime in seconds.')


@backend('synth', 'ppa_vivado', requires='vivado', version_cmd=['vivado', '-version'])
class Measure:
    def ppa_measure(self, params, ctx):
        files = sources(params.files, ctx.cwd)
        require_combinational(files)
        source_hash = fingerprint(files, params.top)
        for path in files:
            if '`include' in path.read_text(encoding='utf-8'):
                raise InvalidInput('include dependencies must be inlined for a complete source fingerprint')
        report = ctx.run.artifacts / 'ppa.txt'
        script = ctx.run.work / 'measure.tcl'
        reads = '\n'.join('read_verilog -sv ' + tcl_path(p) for p in files)
        script.write_text(f'''create_project -in_memory -part {params.device}
{reads}
synth_design -top {params.top} -part {params.device} -flatten_hierarchy rebuilt
set_max_delay {params.constraint_ns} -datapath_only -from [all_inputs] -to [all_outputs]
opt_design
place_design
route_design
set paths [get_timing_paths -from [all_inputs] -to [all_outputs] -max_paths 1]
if {{[llength $paths] == 0}} {{error "No input-to-output timing path"}}
set delay [get_property DATAPATH_DELAY [lindex $paths 0]]
set luts [llength [get_cells -hier -filter {{REF_NAME =~ LUT*}}]]
set ffs [llength [get_cells -hier -filter {{REF_NAME =~ FD*}}]]
set out [open {tcl_path(report)} w]
puts $out "luts=$luts"
puts $out "ffs=$ffs"
puts $out "delay_ns=$delay"
puts $out "version=[version -short]"
close $out
''', encoding='utf-8')
        result = run_process([shutil.which('vivado') or 'vivado', '-mode', 'batch', '-nojournal', '-notrace', '-log', str(ctx.run.work / 'vivado.log'), '-source', str(script)],
                             cwd=ctx.run.work, log_path=ctx.run.artifacts / 'measure.log', timeout_s=params.timeout_s)
        if result.exit_code or result.timed_out or not report.exists():
            return Result(ok=False, data={'exit_code': result.exit_code, 'timed_out': result.timed_out,
                                         'log': ctx.run.handle('measure.log')}, note='No PPA result: Vivado failed, timed out, or did not write its measurement.')
        if fingerprint(files, params.top) != source_hash:
            return Result(ok=False, data={}, note='Sources changed during measurement; result discarded.')
        values = dict(line.split('=', 1) for line in report.read_text().splitlines() if '=' in line)
        record = Record(name=params.name, source_sha256=source_hash, tool='vivado',
                        version=values['version'], part=params.device, constraint_ns=params.constraint_ns,
                        stage='routed_combinational', units={'luts': 'count', 'ffs': 'count', 'delay_ns': 'ns'},
                        metrics={k: float(values[k]) for k in ('luts', 'ffs', 'delay_ns')}, evidence=ctx.run.handle('ppa.txt'))
        return Result(data={'record': record.model_dump()}, note='Routed combinational LUT/FF count and datapath delay. No power estimate, clock Fmax, board measurement, or sequential signoff.')


CATEGORY.ops.append(Op('ppa_measure', MeasureIn, Result, 'Measure routed combinational FPGA area and delay using local Vivado.', long_running=True))
