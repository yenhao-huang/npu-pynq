"""Regression checks for resumed-session accounting in cost analysis."""
import json
import tempfile
import unittest
from pathlib import Path

from analysis.multi_round_analysis import round_usage


class RoundUsageTest(unittest.TestCase):
    def check_rounds(self, events, expected):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number, event in enumerate(events, 1):
                (root / f"round{number}.jsonl").write_text(json.dumps(event) + "\n")
            actual = round_usage({"_dir": root})
            self.assertEqual(actual.keys(), expected.keys())
            for number, (cost, inputs, outputs) in expected.items():
                self.assertAlmostEqual(actual[number][0], cost)
                self.assertEqual(actual[number][1:], (inputs, outputs))

    def test_claude_cost_is_session_total_but_tokens_are_per_round(self):
        self.check_rounds([
            {"type": "result", "total_cost_usd": 0.10,
             "usage": {"input_tokens": 10, "cache_read_input_tokens": 90,
                       "output_tokens": 5}},
            {"type": "result", "total_cost_usd": 0.25,
             "usage": {"input_tokens": 20, "cache_read_input_tokens": 180,
                       "output_tokens": 7}},
        ], {1: (0.10, 100, 5), 2: (0.15, 200, 7)})

    def test_codex_tokens_are_session_totals(self):
        self.check_rounds([
            {"type": "result", "synthesized": "codex",
             "usage": {"input_tokens": 80, "cache_read_input_tokens": 20,
                       "output_tokens": 5}},
            {"type": "result", "synthesized": "codex",
             "usage": {"input_tokens": 150, "cache_read_input_tokens": 50,
                       "output_tokens": 12}},
        ], {1: (0.0, 100, 5), 2: (0.0, 100, 7)})


if __name__ == "__main__":
    unittest.main()
