import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.model.performance import ArrayConfiguration
from src.runtime.benchmark import (
    BENCHMARK_MAGIC,
    BenchmarkError,
    HostMatrixEngine,
    MatrixPoint,
    Provenance,
    build_record,
    compare_int8_to_float,
    compare_records,
    load_record,
    measure_matrix_point,
    publish_record,
    run_matrix_sweep,
)


ARRAY = ArrayConfiguration(rows=8, columns=8, tile_k=256)


def provenance(**overrides):
    values = dict(
        commit="0" * 40,
        branch="npu/issue63-a",
        mode="simulation",
        array=ARRAY,
        requantization_site="host",
        clock_hz=100_000_000.0,
    )
    values.update(overrides)
    return Provenance(**values)


def engine():
    return HostMatrixEngine(max_m=8, max_n=8, max_k=256)


class HostMatrixEngineTests(unittest.TestCase):
    def test_result_is_bit_exact_int32(self):
        a = np.array([[1, -2], [3, 4]], dtype=np.int8)
        b = np.array([[5, 6], [-7, 8]], dtype=np.int8)
        product = engine().run(a, b)
        self.assertEqual(product.dtype, np.int32)
        np.testing.assert_array_equal(
            product, a.astype(np.int32) @ b.astype(np.int32)
        )

    def test_modeled_cycles_match_the_documented_serialized_minimum(self):
        # docs/manual/hw/matrx_controller.md section 7.
        self.assertEqual(
            HostMatrixEngine.modeled_cycles(2, 2, 4),
            2 * 4 + 4 * 2 + 2 * 2 + 4 + 2 + 2 + 1,
        )

    def test_non_int8_operands_and_oversized_jobs_are_rejected(self):
        device = engine()
        with self.assertRaises(BenchmarkError):
            device.run(np.ones((2, 2), dtype=np.int16), np.ones((2, 2), dtype=np.int8))
        with self.assertRaises(BenchmarkError):
            device.run(
                np.ones((9, 2), dtype=np.int8), np.ones((2, 2), dtype=np.int8)
            )


class MeasurementTests(unittest.TestCase):
    def test_point_reports_measured_and_modeled_cycles(self):
        record = measure_matrix_point(
            engine(), MatrixPoint(8, 8, 64), array=ARRAY, repeat_count=2
        )
        self.assertEqual(record["shape"], "8x8x64")
        self.assertEqual(
            record["cycles"]["measured"], HostMatrixEngine.modeled_cycles(8, 8, 64)
        )
        self.assertGreater(record["cycles"]["modeled_compute"], 0)
        self.assertGreater(record["cycles"]["ratio"], 1.0)

    def test_load_and_output_dominate_the_serialized_job(self):
        # The analytical model counts the compute wavefront only, so a job whose
        # loads cannot be hidden costs far more than the model suggests. This is
        # the gap a ping-pong buffer is meant to close.
        record = measure_matrix_point(
            engine(), MatrixPoint(8, 8, 128), array=ARRAY, repeat_count=2
        )
        self.assertGreater(record["cycles"]["ratio"], 5.0)

    def test_wrong_result_fails_instead_of_being_timed(self):
        class WrongEngine:
            max_m = max_n = max_k = 256
            last_metrics = None

            def run(self, a, b, **_):
                return np.zeros((a.shape[0], b.shape[1]), dtype=np.int32)

        with self.assertRaises(BenchmarkError):
            measure_matrix_point(
                WrongEngine(), MatrixPoint(2, 2, 2), array=ARRAY, repeat_count=2
            )

    def test_unrepeatable_result_is_rejected(self):
        class DriftingEngine:
            max_m = max_n = max_k = 256

            def __init__(self):
                self.calls = 0
                self.last_metrics = None

            def run(self, a, b, **_):
                self.calls += 1
                product = a.astype(np.int32) @ b.astype(np.int32)
                if self.calls > 1:
                    product = product + 1
                return product

        with self.assertRaises(BenchmarkError):
            measure_matrix_point(
                DriftingEngine(), MatrixPoint(2, 2, 2), array=ARRAY, repeat_count=2
            )

    def test_missing_cycle_telemetry_is_reported_as_unavailable(self):
        class SilentEngine:
            max_m = max_n = max_k = 256
            last_metrics = None

            def run(self, a, b, **_):
                return a.astype(np.int32) @ b.astype(np.int32)

        record = measure_matrix_point(
            SilentEngine(), MatrixPoint(2, 2, 2), array=ARRAY, repeat_count=2
        )
        self.assertIsNone(record["cycles"]["measured"])
        self.assertIsNone(record["cycles"]["ratio"])

    def test_single_repetition_is_rejected(self):
        with self.assertRaises(BenchmarkError):
            measure_matrix_point(
                engine(), MatrixPoint(2, 2, 2), array=ARRAY, repeat_count=1
            )

    def test_sweep_rejects_duplicate_shapes(self):
        with self.assertRaises(BenchmarkError):
            run_matrix_sweep(
                engine(), [MatrixPoint(2, 2, 2), MatrixPoint(2, 2, 2)], array=ARRAY
            )


class AccuracyTests(unittest.TestCase):
    def test_identical_predictions_agree(self):
        baseline = np.array([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])
        quantized = np.array([[0, 2, 1], [3, 0, 1]], dtype=np.int8)
        report = compare_int8_to_float(quantized, baseline, scale=1.0, top_k=(1, 2))
        self.assertEqual(report["top1_prediction_agreement"], 1.0)
        self.assertEqual(report["set_agreement"]["top1"], 1.0)
        self.assertEqual(report["logit_difference"]["max_absolute"], 0.0)

    def test_quantization_that_flips_the_prediction_is_visible(self):
        baseline = np.array([[1.0, 1.2]])
        quantized = np.array([[2, 1]], dtype=np.int8)
        report = compare_int8_to_float(quantized, baseline, scale=1.0, top_k=(1,))
        self.assertEqual(report["top1_prediction_agreement"], 0.0)
        self.assertGreater(report["logit_difference"]["max_absolute"], 0.0)

    def test_invalid_inputs_are_rejected(self):
        baseline = np.array([[1.0, 2.0]])
        with self.assertRaises(BenchmarkError):
            compare_int8_to_float(np.array([[1, 2]]), baseline, scale=1.0, top_k=(1,))
        with self.assertRaises(BenchmarkError):
            compare_int8_to_float(
                np.array([[1, 2]], dtype=np.int8), baseline, scale=0.0, top_k=(1,)
            )
        with self.assertRaises(BenchmarkError):
            compare_int8_to_float(
                np.array([[1, 2]], dtype=np.int8), baseline, scale=1.0, top_k=(3,)
            )


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.sweep = run_matrix_sweep(
            engine(), [MatrixPoint(8, 8, 64)], array=ARRAY, repeat_count=2
        )

    def test_record_round_trips_as_canonical_json(self):
        record = build_record(
            provenance=provenance(), matrix_sweep=self.sweep, cycles_measured=False
        )
        path = self.directory / "record.json"
        encoded = publish_record(path, record)
        self.assertEqual(path.read_bytes(), encoded)
        reloaded = load_record(path)
        self.assertEqual(reloaded["magic"], BENCHMARK_MAGIC)
        self.assertEqual(
            reloaded["provenance"]["requantization_site"], "host"
        )
        self.assertFalse(reloaded["cycles_measured"])

    def test_a_simulation_record_may_not_claim_measured_cycles(self):
        with self.assertRaises(BenchmarkError):
            build_record(
                provenance=provenance(mode="simulation"),
                matrix_sweep=self.sweep,
                cycles_measured=True,
            )

    def test_empty_record_is_rejected(self):
        with self.assertRaises(BenchmarkError):
            build_record(provenance=provenance(), cycles_measured=False)

    def test_unknown_mode_and_requantization_site_are_rejected(self):
        with self.assertRaises(BenchmarkError):
            provenance(mode="emulator")
        with self.assertRaises(BenchmarkError):
            provenance(requantization_site="somewhere")

    def test_non_record_file_is_rejected(self):
        path = self.directory / "other.json"
        path.write_text(json.dumps({"magic": "SOMETHING_ELSE"}), encoding="utf-8")
        with self.assertRaises(BenchmarkError):
            load_record(path)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.sweep = run_matrix_sweep(
            engine(), [MatrixPoint(8, 8, 64)], array=ARRAY, repeat_count=2
        )

    def record(self, **overrides):
        return build_record(
            provenance=provenance(**overrides),
            matrix_sweep=self.sweep,
            cycles_measured=False,
        )

    def test_matching_provenance_compares(self):
        report = compare_records(self.record(), self.record(branch="npu/issue64-a"))
        self.assertTrue(report["comparable"])
        self.assertEqual(report["shapes"][0]["shape"], "8x8x64")
        self.assertEqual(report["shapes"][0]["cycles"]["speedup"], 1.0)

    def test_different_array_blocks_the_comparison(self):
        other = ArrayConfiguration(rows=2, columns=2, tile_k=256)
        report = compare_records(self.record(), self.record(array=other))
        self.assertFalse(report["comparable"])
        self.assertIn("array", report["blocking_differences"])

    def test_different_requantization_site_blocks_the_comparison(self):
        report = compare_records(
            self.record(), self.record(requantization_site="hardware")
        )
        self.assertFalse(report["comparable"])
        self.assertIn("requantization_site", report["blocking_differences"])

    def test_shapes_present_in_only_one_record_are_named(self):
        wider = run_matrix_sweep(
            engine(), [MatrixPoint(8, 8, 128)], array=ARRAY, repeat_count=2
        )
        after = build_record(
            provenance=provenance(), matrix_sweep=wider, cycles_measured=False
        )
        report = compare_records(self.record(), after)
        self.assertEqual(report["only_in_before"], ["8x8x64"])
        self.assertEqual(report["only_in_after"], ["8x8x128"])

    def test_non_record_input_is_rejected(self):
        with self.assertRaises(BenchmarkError):
            compare_records({"magic": "nope"}, self.record())


if __name__ == "__main__":
    unittest.main()
