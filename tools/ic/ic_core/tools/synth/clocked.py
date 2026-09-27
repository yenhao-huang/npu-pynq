"""Clocked out-of-context implementation with explicit throughput assumptions."""
import math
import re
import shutil
from typing import Literal
from pydantic import Field
from ...errors import InvalidInput
from ...process import run as run_process
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, Identifier, fingerprint, sources, tcl_path
from . import CATEGORY, DEFAULT_PART
from .exploration import Result


class ClockedIn(ExplorationInput):
    backend: str | None = Field(default='clocked_vivado', description='Operation-specific execution backend.')
    files: list[str] = Field(min_length=1, description='Self-contained synthesizable SystemVerilog sources.')
    top: Identifier = Field(description='Top module identifier, identical in separately compiled variants.')
    clock_port: Identifier = Field(default='clk', description='Single clock input port.')
    name: str = Field(min_length=1, description='Measurement candidate identifier.')
    device: str = Field(default=DEFAULT_PART, pattern=r'^[A-Za-z0-9_-]+$', description='Target FPGA part.')
    period_ns: float = Field(default=10, gt=0, allow_inf_nan=False, description='Requested clock period in nanoseconds.')
    latency_cycles: int = Field(ge=1, description='Caller-verified transaction latency; timing analysis does not verify protocol.')
    initiation_interval: int = Field(ge=1, description='Caller-verified sustained cycles per transaction.')
    directive: Literal['Default', 'Explore'] = Field(default='Default', description='Matched placement and routing directive.')
    timeout_s: float = Field(default=900, gt=0, le=3600, allow_inf_nan=False, description='Maximum process runtime in seconds.')


def utilization(text):
    aliases = {'Slice LUTs':'luts', 'CLB LUTs':'luts', 'Slice Registers':'ffs',
               'CLB Registers':'ffs', 'Block RAM Tile':'brams', 'DSPs':'dsps',
               'LUT as Memory':'lut_memory', 'LUT as Logic':'lut_logic'}
    values = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split('|')]
        if len(parts)<3:
            continue
        label = parts[1].rstrip('*').strip()
        if label in aliases and re.fullmatch(r'[0-9,]+(?:\.[0-9]+)?', parts[2]):
            values[aliases[label]] = float(parts[2].replace(',', ''))
    if not {'luts','ffs','brams','dsps'} <= values.keys():
        raise ValueError('utilization report lacks required resource classes')
    return values


@backend('synth', 'clocked_vivado', requires='vivado', version_cmd=['vivado', '-version'])
class ClockedMeasure:
    def clocked_ppa(self, params, ctx):
        files = sources(params.files, ctx.cwd)
        if any('`include' in p.read_text(encoding='utf-8') for p in files):
            raise InvalidInput('inline include dependencies for complete source identity')
        source_hash = fingerprint(files, params.top)
        metrics = ctx.run.artifacts/'clocked.txt'
        util = ctx.run.artifacts/'utilization.txt'
        timing = ctx.run.artifacts/'timing.txt'
        script = ctx.run.work/'clocked.tcl'
        reads = '\n'.join('read_verilog -sv '+tcl_path(p) for p in files)
        script.write_text(f'''create_project -in_memory -part {params.device}
{reads}
synth_design -mode out_of_context -top {params.top} -part {params.device} -flatten_hierarchy rebuilt
if {{[llength [get_ports {params.clock_port}]] != 1}} {{error "Clock port missing"}}
create_clock -name ic_clock -period {params.period_ns} [get_ports {params.clock_port}]
opt_design
place_design -directive {params.directive}
route_design -directive {params.directive}
set paths [get_timing_paths -delay_type max -from [all_registers -clock ic_clock] -to [all_registers -clock ic_clock] -max_paths 1]
if {{[llength $paths] != 1}} {{error "No register-to-register setup timing path"}}
set path [lindex $paths 0]
report_utilization -file {tcl_path(util)}
report_timing -of_objects $paths -file {tcl_path(timing)}
set out [open {tcl_path(metrics)} w]
puts $out "slack_ns=[get_property SLACK $path]"
puts $out "requirement_ns=[get_property REQUIREMENT $path]"
puts $out "datapath_ns=[get_property DATAPATH_DELAY $path]"
puts $out "version=[version -short]"
puts $out "build=[string map {{\\n | \\r {{}}}} [version]]"
close $out
''', encoding='utf-8')
        result = run_process([shutil.which('vivado') or 'vivado', '-mode','batch','-nojournal','-notrace','-log',str(ctx.run.work/'vivado.log'),'-source',str(script)], cwd=ctx.run.work, log_path=ctx.run.artifacts/'clocked.log', timeout_s=params.timeout_s)
        log = ctx.run.handle('clocked.log')
        if result.exit_code or result.timed_out or not metrics.exists() or not util.exists():
            return Result(ok=False, data=dict(log=log, exit_code=result.exit_code, timed_out=result.timed_out), note='Implementation failed or timed out; no clocked PPA result.')
        if source_hash != fingerprint(files, params.top):
            return Result(ok=False, data={'log':log}, note='Sources changed; measurement discarded.')
        try:
            raw = dict(line.split('=',1) for line in metrics.read_text().splitlines() if '=' in line)
            slack, requirement = float(raw['slack_ns']), float(raw['requirement_ns'])
            critical_period = params.period_ns-slack
            if not math.isfinite(critical_period) or critical_period<=0 or not math.isclose(requirement,params.period_ns,abs_tol=.01):
                raise ValueError('unsupported timing requirement or invalid critical period')
            resources = utilization(util.read_text())
        except (ValueError, KeyError) as error:
            return Result(ok=False, data={'log':log}, note=f'Unusable measurement: {error}')
        fmax = 1000/critical_period
        record = dict(name=params.name, source_sha256=source_hash, tool='vivado', version=raw['version'], build=raw['build'],
                      part=params.device, stage='routed_clocked_ooc', period_ns=params.period_ns, directive=params.directive,
                      latency_cycles=params.latency_cycles, initiation_interval=params.initiation_interval,
                      metrics=dict(resources, slack_ns=slack, critical_period_ns=critical_period,
                                   estimated_fmax_mhz=fmax, throughput_mtransactions_s=fmax/params.initiation_interval),
                      evidence={'metrics':ctx.run.handle(metrics.name),'utilization':ctx.run.handle(util.name),'timing':ctx.run.handle(timing.name),'log':log})
        return Result(data={'record':record}, note='Routed single-clock register-to-register timing estimate; latency and initiation interval are caller-verified. Excludes top-level I/O timing and board validation; no power claim.')


CATEGORY.ops.append(Op('clocked_ppa', ClockedIn, Result, 'Measure routed clocked FPGA resources and normalize estimated throughput by initiation interval.', long_running=True))
