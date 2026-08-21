import unittest

from src.test.model.performance import (
    DEFAULT_TARGET,
    ArrayConfiguration,
    PerformanceAssumptions,
    ResourceEstimate,
    TargetResources,
    assess_cycle_measurement,
    assess_resources,
    estimate_matmul,
)


class AccountingTests(unittest.TestCase):
    def test_specification_operation_and_payload_example(self):
        report = estimate_matmul(
            m=2,
            n=3,
            k=4,
            array=ArrayConfiguration(rows=2, columns=3, tile_k=4),
            assumptions=PerformanceAssumptions(launch_overhead_seconds=0.0),
        )
        self.assertEqual(report.operations, 48)
        self.assertEqual(report.payload_bytes, 44)
        self.assertEqual(report.compute_cycles, 7)
        self.assertEqual(report.tile_count, 1)

    def test_non_aligned_tiles_include_each_edge_fill_and_drain(self):
        report = estimate_matmul(
            m=3,
            n=3,
            k=5,
            array=ArrayConfiguration(rows=2, columns=2, tile_k=4),
        )
        self.assertEqual(report.tile_count, 8)
        self.assertEqual(report.compute_cycles, 28)
        self.assertGreater(report.array_utilization, 0.0)
        self.assertLessEqual(report.array_utilization, 1.0)

    def test_invalid_dimensions_or_configuration_are_rejected(self):
        with self.assertRaises(ValueError):
            estimate_matmul(0, 1, 1, ArrayConfiguration(1, 1, 1))
        with self.assertRaises(ValueError):
            ArrayConfiguration(0, 1, 1)
        with self.assertRaises(ValueError):
            PerformanceAssumptions(sustained_bandwidth_bytes_per_second=0)


class RooflineTests(unittest.TestCase):
    def test_bandwidth_bound_job(self):
        report = estimate_matmul(
            2,
            3,
            4,
            ArrayConfiguration(2, 3, 4),
            PerformanceAssumptions(launch_overhead_seconds=0.0),
        )
        self.assertEqual(report.limiting_factor, "bandwidth")
        self.assertEqual(report.modeled_seconds, report.transport_seconds)

    def test_compute_bound_job_and_launch_overhead(self):
        assumptions = PerformanceAssumptions(
            sustained_bandwidth_bytes_per_second=100_000_000_000.0,
            launch_overhead_seconds=2e-6,
        )
        report = estimate_matmul(2, 3, 4, ArrayConfiguration(2, 3, 4), assumptions)
        self.assertEqual(report.limiting_factor, "compute")
        self.assertAlmostEqual(
            report.modeled_seconds, report.compute_seconds + 2e-6
        )
        self.assertAlmostEqual(
            report.operations_per_second,
            report.operations / report.modeled_seconds,
        )

    def test_reports_are_deterministic(self):
        arguments = dict(
            m=7,
            n=5,
            k=9,
            array=ArrayConfiguration(4, 4, 8),
            assumptions=PerformanceAssumptions(),
        )
        self.assertEqual(estimate_matmul(**arguments), estimate_matmul(**arguments))


class ResourceTests(unittest.TestCase):
    def test_default_target_matches_xc7z020_contract(self):
        self.assertEqual(
            DEFAULT_TARGET,
            TargetResources(luts=53_200, flip_flops=106_400, bram36=140, dsp48=220),
        )

    def test_values_at_budget_pass_and_values_above_fail(self):
        target = TargetResources(luts=100, flip_flops=200, bram36=20, dsp48=140)
        passing = assess_resources(
            ResourceEstimate(luts=75, flip_flops=150, bram36=15, dsp48=105),
            target,
        )
        failing = assess_resources(
            ResourceEstimate(luts=76, flip_flops=150, bram36=15, dsp48=106),
            target,
        )
        self.assertTrue(passing.passed)
        self.assertEqual(passing.utilization_percent["dsp48"], 75.0)
        self.assertFalse(failing.passed)
        self.assertEqual(set(failing.over_budget), {"luts", "dsp48"})

    def test_invalid_resource_values_are_rejected(self):
        with self.assertRaises(ValueError):
            ResourceEstimate(luts=-1, flip_flops=0, bram36=0, dsp48=0)
        with self.assertRaises(ValueError):
            TargetResources(luts=0, flip_flops=1, bram36=1, dsp48=1)


class MeasurementTests(unittest.TestCase):
    def test_inclusive_ten_percent_cycle_gate(self):
        self.assertTrue(assess_cycle_measurement(1000, 900).passed)
        self.assertTrue(assess_cycle_measurement(1000, 1100).passed)
        self.assertFalse(assess_cycle_measurement(1000, 899).passed)
        self.assertFalse(assess_cycle_measurement(1000, 1101).passed)

    def test_measurement_report_records_delta(self):
        report = assess_cycle_measurement(1000, 950)
        self.assertEqual(report.delta_cycles, -50)
        self.assertEqual(report.absolute_error_percent, 5.0)


if __name__ == "__main__":
    unittest.main()
