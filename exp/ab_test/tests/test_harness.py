import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.config import Config
from src.problems.catalog import Problem, discover, reference_source
from src.agent.context import Context
from src.agent.llm import ClaudeModel, ModelResponse
from src.agent.rounds import run_rounds
from src.agent.parse import summarize
from src.grade.heldout import grade_heldout
from src.grade.shipped import grade_shipped
from src.logging.recorder import Recorder
from src.runner import attempt

ROOT = Path(__file__).resolve().parents[1]

class FakeModel:
    def __init__(self, fail_at=None, timeout_at=None):
        self.calls = []
        self.fail_at = fail_at
        self.timeout_at = timeout_at

    def generate(self, prompt, arm, work, flags):
        self.calls.append((prompt, arm, flags))
        number = len(self.calls)
        (work / 'solution.v').write_text(f'// round {number}')
        stdout = '\n'.join(json.dumps(event) for event in [
            {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': f'answer {number}'}]}},
            {'type': 'result', 'is_error': number == self.fail_at,
             'result': 'API error' if number == self.fail_at else 'done',
             'total_cost_usd': 0.25, 'num_turns': 1, 'usage': {'input_tokens': 10}},
        ])
        return ModelResponse(stdout, 1.0, number == self.timeout_at)

class HarnessTests(unittest.TestCase):
    def test_legacy_entry_point_and_package_entry_point(self):
        import sys
        for args in ([str(ROOT/'src/ab_test_single_round.py'), '--help'], ['-m', 'src.main', '--help']):
            result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--rounds', result.stdout)
            self.assertIn('--repetitions', result.stdout)
            self.assertIn('--effort', result.stdout)

    def test_single_and_multi_round_context(self):
        for count in (1, 3):
            with self.subTest(rounds=count), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                recorder = Recorder(work / 'logs')
                model = FakeModel()
                context = Context('specification', 'notools', count)
                records = run_rounds(context, model, work, recorder)
                self.assertEqual(len(records), count)
                self.assertEqual(records[-1]['solution'], f'// round {count}')
                self.assertEqual(len(json.loads((work/'logs/context.json').read_text())['messages']), count*2)
                if count == 1:
                    self.assertEqual(model.calls[0][2], ['--no-session-persistence'])
                else:
                    self.assertEqual(model.calls[0][2], ['--session-id', context.session_id])
                    self.assertEqual(model.calls[1][2], ['--resume', context.session_id])
                    self.assertIn('Round 2 of 3', model.calls[1][0])
                summary = summarize([r['response'].stdout for r in records])
                self.assertEqual(summary['cost_usd'], count * 0.25)
                self.assertEqual(summary['usage']['input_tokens'], count * 10)

    def test_errors_stop_rounds_and_preserve_logs(self):
        for model in (FakeModel(fail_at=2), FakeModel(timeout_at=2)):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                records = run_rounds(Context('spec', 'tools', 3), model, work, Recorder(work/'logs'))
                self.assertEqual(len(records), 2)
                self.assertIsNotNone(records[-1]['error'])
                self.assertTrue((work/'logs/round2.jsonl').exists())

    def test_grading_follows_conversation_and_snapshots_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'design_description.txt').write_text('spec')
            problem = Problem('demo', root, 'demo', root/'ref.v', 'test')
            model = FakeModel()
            seen = []
            class Grader:
                def evaluate(self, problem, solution, work):
                    self_count = len(model.calls)
                    if self_count != 3:
                        raise AssertionError('grading leaked into conversation')
                    seen.append(solution.read_text())
                    return dict(passed=True, passed_shipped=True, passed_heldout=True, ppa_ok=True,
                                area_um2=len(seen), power_w=4-len(seen), delay_ns=2, wns_ns=-1)
            row = attempt(problem, 'notools', 0, Config(rounds=3), model, Grader(), root/'output')
            self.assertEqual(seen, ['// round 1', '// round 2', '// round 3'])
            self.assertTrue(row['passed'])
            self.assertEqual(len(row['pareto_frontier']), 3)
            self.assertEqual(row['final_round']['round'], 3)
            for candidate in row['pareto_frontier']:
                self.assertTrue((root/'output'/candidate['solution_path']).exists())
            self.assertEqual(len(row['per_round']), 3)

    def test_failed_model_round_is_not_graded_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'design_description.txt').write_text('spec')
            problem = Problem('demo', root, 'demo', root/'ref.v', 'test')
            class Grader:
                def evaluate(self, *args):
                    raise AssertionError('must not grade a failed model response')
            row = attempt(problem, 'notools', 0, Config(rounds=3), FakeModel(fail_at=1), Grader(), root/'output')
            self.assertFalse(row['passed'])
            self.assertEqual(row['failure'], 'API error')

    def test_adapter_session_tools_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with patch('src.agent.llm.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '{}', '')) as run:
                ClaudeModel(Config(rounds=3)).generate('prompt', 'notools', work, ['--resume', 'session'])
                cmd = run.call_args.args[0]
                self.assertIn('--resume', cmd)
                self.assertEqual(cmd[cmd.index('--effort') + 1], 'medium')
                self.assertIn('Write,Read', cmd)
                self.assertNotIn('--no-session-persistence', cmd)
                self.assertIn(f'Read({Config().problems}/**)', cmd)
            with patch('src.agent.llm.subprocess.run', side_effect=subprocess.TimeoutExpired('claude', 1, output=b'partial')):
                response = ClaudeModel(Config()).generate('prompt', 'tools', work, ['--no-session-persistence'])
                self.assertTrue(response.timed_out)
                self.assertEqual(response.stdout, 'partial')
                self.assertTrue((work/'mcp.json').exists())

    def test_real_verilog_grading_passes_reference_and_rejects_wrong_design(self):
        problem = discover(ROOT/'problems')['adder_8bit']
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            solution = work/'solution.v'
            source = reference_source(problem)
            solution.write_text(source)
            self.assertTrue(grade_shipped(problem, solution, work)[0])
            self.assertTrue(grade_heldout(problem, solution, work)[0])
            solution.write_text('module adder_8bit(input [7:0] a, b, input cin, output [7:0] sum, output cout); assign sum = 0; assign cout = 0; endmodule')
            self.assertFalse(grade_shipped(problem, solution, work)[0])
            self.assertFalse(grade_heldout(problem, solution, work)[0])

if __name__ == '__main__':
    unittest.main()
