import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from src.grade.pareto import frontier, grouped_frontiers, top3_pdap
from src.grade.ppa import measure_ppa
from src.grade.evaluator import Grader
from src.grade.shipped import grade_shipped
from src.problems.catalog import discover, reference_source
from src.config import Config


def candidate(name, area=2, power=2, delay=2, **extra):
    return dict(name=name, area_um2=area, power_w=power, delay_ns=delay,
                passed=True, passed_shipped=True, passed_heldout=True, ppa_ok=True, **extra)

class ParetoTests(unittest.TestCase):
    def test_pdap_ranks_only_frontier_and_keeps_full_frontier(self):
        points = [candidate('A', 1, 5, 5), candidate('B', 2, 2, 2),
                  candidate('C', 3, 2, 2), candidate('D', 4, 1, 1)]
        ranked = top3_pdap(points)
        self.assertEqual([c['name'] for c in ranked], ['D', 'B', 'A'])
        self.assertEqual([c['pdap'] for c in ranked], [4, 8, 25])
        self.assertEqual([c['name'] for c in frontier(points)], ['A', 'B', 'D'])
        self.assertNotIn('pdap', points[0])
        self.assertEqual(top3_pdap([]), [])
        self.assertEqual(len(top3_pdap([candidate('only')])), 1)

    def test_tradeoffs_dominance_ties_and_wns(self):
        points = [candidate('area', 1, 4, 4), candidate('power', 4, 1, 4),
                  candidate('delay', 4, 4, 1), candidate('balanced'),
                  candidate('dominated', 3, 3, 3), candidate('tie', wns_ns=-100)]
        self.assertEqual([c['name'] for c in frontier(points)],
                         ['area', 'power', 'delay', 'balanced', 'tie'])
        self.assertEqual(frontier([]), [])

    def test_invalid_measurements_and_failed_checks_are_excluded(self):
        points = [candidate('valid')]
        for key, value in [('power_w', None), ('power_w', 0), ('power_w', float('nan')),
                           ('delay_ns', float('inf')), ('area_um2', -1), ('ppa_ok', False),
                           ('passed', False), ('passed_heldout', None), ('delay_ns', True)]:
            point = candidate('invalid')
            point[key] = value
            points.append(point)
        self.assertEqual([c['name'] for c in frontier(points)], ['valid'])
        skipped = candidate('known skip')
        skipped.update(passed_heldout=None, heldout_skipped=True)
        self.assertEqual(frontier([skipped]), [skipped])

    def test_grouping_does_not_compare_different_problems(self):
        rows = [dict(problem='p', arm='tools', rep=0, per_round=[candidate('large', 3, 3, 3)]),
                dict(problem='p', arm='notools', rep=0, per_round=[candidate('small')]),
                dict(problem='q', arm='tools', rep=0, per_round=[candidate('other', 100, 100, 100)])]
        result = grouped_frontiers(rows)
        self.assertEqual([c['name'] for c in result['p']['frontier']], ['small'])
        self.assertEqual(result['p']['by_arm']['tools'][0]['name'], 'large')
        self.assertEqual(result['q']['frontier'][0]['name'], 'other')

    def test_delay_derived_from_fmax_and_wns_preserved(self):
        data = dict(area={'cell_area_um2': 10}, timing={'fmax_mhz': 400, 'wns_ns': -0.5},
                    power={'total_w': .01}, ok=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sol = root/'solution.v'
            sol.write_text('module demo; endmodule')
            with patch('src.grade.ppa.ports_of', return_value=[]), patch('src.grade.ppa.subprocess.run',
                    return_value=SimpleNamespace(stdout=json.dumps(data), stderr='', returncode=0)):
                result = measure_ppa(SimpleNamespace(top='demo'), sol, root, Config())
            self.assertEqual(result['delay_ns'], 2.5)
            self.assertEqual(result['fmax_mhz'], 400)
            self.assertEqual(result['wns_ns'], -.5)
            self.assertNotIn('adp', result)

    def test_port_extraction_failure_is_not_a_correctness_pass(self):
        with patch('src.grade.evaluator.grade_shipped', return_value=(True, '')), \
             patch('src.grade.evaluator.grade_heldout', return_value=(None, 'port extraction failed')), \
             patch('src.grade.evaluator.measure_ppa') as ppa:
            result = Grader(Config()).evaluate(SimpleNamespace(id='demo'), None, None)
            self.assertFalse(result['passed'])
            ppa.assert_not_called()

    def test_known_testbench_patches_are_applied(self):
        root = Path(__file__).resolve().parents[1]
        problems = discover(root/'problems')
        for pid in ('clkgenerator', 'ring_counter'):
            with self.subTest(pid=pid), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                sol = work/'solution.v'
                sol.write_text(reference_source(problems[pid]))
                passed, info = grade_shipped(problems[pid], sol, work)
                self.assertTrue(passed, info)

if __name__ == '__main__':
    unittest.main()
