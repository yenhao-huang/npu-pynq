"""Local tool discovery must work without the optional HTTP daemon packages."""

import json
import subprocess
import sys
from pathlib import Path


def test_catalogue_without_daemon_dependencies():
    code = """
import importlib.abc, runpy, sys
class NoDaemon(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ic_daemon', 'fastapi', 'uvicorn'}:
            raise ImportError('Optional daemon dependency imported: ' + fullname)
sys.meta_path.insert(0, NoDaemon())
sys.argv = ['ic', 'tools', '--compact']
runpy.run_module('ic_cli.main', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=True,
    )
    tools = json.loads(result.stdout)['tools']
    assert {tool['name'] for tool in tools} == {
        'lint', 'sim', 'signals', 'first_mismatch', 'value_at',
        'value_range', 'show_wave', 'synth',
        'optimization_rules', 'width_advice', 'ppa_measure', 'ppa_provenance',
        'ppa_compare', 'ppa_pareto', 'ppa_reward', 'ppa_select', 'comb_check', 'rtl_evaluate',
        'clocked_ppa', 'vector_equivalence', 'yosys_equivalence', 'synth_adder_tree',
        'synth_popcount', 'synth_priority_encoder', 'synth_leading_zero', 'synth_barrel_shifter',
        'synth_onehot_mux', 'synth_argmax_tree', 'architecture_sweep',
        'synth_fifo', 'sequential_scoreboard',
        'synth_prefix_adder', 'synth_csd_multiplier', 'synth_mcm',
        'synth_constant_modulo',
        'synth_fir', 'synth_dot_product',
        'synth_saturating_alu',
        'synth_crc_parallel', 'synth_lfsr_jump',
        'gf2_equivalence',
        'synth_serial_multiplier', 'synth_divider', 'latency_throughput', 'synth_booth_multiplier', 'synth_fsm', 'timing_constraint_audit', 'synth_banked_regfile', 'synth_skid_buffer',
        'resource_tradeoff', 'paired_repeat_summary',
        'netlist_profile', 'critical_cone', 'fanout_analysis', 'memory_inference',
    }
