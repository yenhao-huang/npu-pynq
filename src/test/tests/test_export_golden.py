"""Every export frontend still builds the ModelGraph recorded in
src/test/vectors/export_golden.json, bit for bit."""

import json
import unittest

from src.test.vectors import generate_export_golden as golden


class ExportGoldenTest(unittest.TestCase):
    def test_frontends_match_recorded_digests(self):
        recorded = json.loads(golden.GOLDEN.read_text())
        for name in golden.CASES:
            with self.subTest(name):
                result = golden.run_case(name)
                if result is None:
                    self.skipTest(f'{name}: inputs unavailable')
                self.assertEqual(result, recorded[name])


if __name__ == '__main__':
    unittest.main()
