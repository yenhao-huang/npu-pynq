"""Tests for the ImageNet preprocessing and label-decoding contract."""

from __future__ import annotations

import unittest

import numpy as np

from src.export.imagenet import (
    CROP_SIZE,
    ImageNetInputError,
    decode_logits,
    dequantized_preview,
    input_scale,
    load_class_names,
    normalize_crop,
    preprocessing_contract,
    quantize_input,
    resolve_class_index,
    scale_from_quantization,
)


def make_manifest(multiplier: int = 44640607, shift: int = 0) -> dict:
    return {
        "tensors": [
            {
                "layout": "NHWC",
                "name": "input",
                "quantization": {
                    "multiplier_q31": multiplier,
                    "shift": shift,
                    "zero_point": 0,
                },
                "shape": [1, CROP_SIZE, CROP_SIZE, 3],
            }
        ]
    }


class ScaleTest(unittest.TestCase):
    def test_q31_scale_round_trips_through_the_manifest(self):
        scale = scale_from_quantization(44640607, 0)
        self.assertAlmostEqual(scale, 44640607 / (1 << 31))
        self.assertEqual(input_scale(make_manifest()), scale)

    def test_shift_multiplies_the_encoded_ratio(self):
        self.assertAlmostEqual(
            scale_from_quantization(1 << 30, 2), 2.0, places=12
        )

    def test_invalid_quantization_is_rejected(self):
        for multiplier, shift in ((0, 0), (1 << 31, 0), (1, 32), (1, -1)):
            with self.assertRaises(ImageNetInputError):
                scale_from_quantization(multiplier, shift)

    def test_missing_or_offset_tensor_is_rejected(self):
        with self.assertRaises(ImageNetInputError):
            input_scale(make_manifest(), "absent")
        offset = make_manifest()
        offset["tensors"][0]["quantization"]["zero_point"] = 1
        with self.assertRaises(ImageNetInputError):
            input_scale(offset)


class NormalizeAndQuantizeTest(unittest.TestCase):
    def setUp(self):
        self.scale = scale_from_quantization(44640607, 0)
        rng = np.random.default_rng(20260920)
        self.crop = rng.integers(
            0, 256, size=(CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8
        )

    def test_normalization_matches_the_recorded_contract(self):
        contract = preprocessing_contract()
        normalized = normalize_crop(self.crop)
        self.assertEqual(normalized.shape, (1, 3, CROP_SIZE, CROP_SIZE))
        self.assertEqual(normalized.dtype, np.float32)
        mean = np.array(contract["channel_mean"], dtype=np.float32)
        deviation = np.array(
            contract["channel_standard_deviation"], dtype=np.float32
        )
        expected = (
            self.crop.astype(np.float32) / np.float32(255.0) - mean
        ) / deviation
        np.testing.assert_allclose(
            normalized[0], expected.transpose(2, 0, 1), rtol=0, atol=0
        )

    def test_quantization_is_symmetric_int8_and_deterministic(self):
        normalized = normalize_crop(self.crop)
        first = quantize_input(normalized, self.scale)
        second = quantize_input(normalized, self.scale)
        self.assertEqual(first.dtype, np.int8)
        self.assertEqual(first.shape, (1, CROP_SIZE, CROP_SIZE, 3))
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(int(first.min()), -127)
        self.assertLessEqual(int(first.max()), 127)

    def test_quantization_rounds_half_away_from_zero_and_saturates(self):
        normalized = np.zeros((1, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        normalized[0, 0, 0, 0] = np.float32(0.5 * self.scale)
        normalized[0, 0, 0, 1] = np.float32(-0.5 * self.scale)
        normalized[0, 0, 0, 2] = np.float32(1000.0)
        normalized[0, 0, 0, 3] = np.float32(-1000.0)
        quantized = quantize_input(normalized, self.scale)
        self.assertEqual(int(quantized[0, 0, 0, 0]), 1)
        self.assertEqual(int(quantized[0, 0, 1, 0]), -1)
        self.assertEqual(int(quantized[0, 0, 2, 0]), 127)
        self.assertEqual(int(quantized[0, 0, 3, 0]), -127)

    def test_preview_recovers_the_crop_within_one_quantization_step(self):
        quantized = quantize_input(normalize_crop(self.crop), self.scale)
        preview = dequantized_preview(quantized, self.scale)
        self.assertEqual(preview.dtype, np.uint8)
        self.assertEqual(preview.shape, (CROP_SIZE, CROP_SIZE, 3))
        deviation = np.array(
            preprocessing_contract()["channel_standard_deviation"],
            dtype=np.float32,
        )
        tolerance = float(np.max(self.scale * deviation * 255.0)) + 1.0
        difference = np.abs(
            preview.astype(np.int16) - self.crop.astype(np.int16)
        )
        self.assertLessEqual(float(difference.max()), tolerance)

    def test_malformed_inputs_are_rejected(self):
        with self.assertRaises(ImageNetInputError):
            normalize_crop(np.zeros((10, 10, 3), dtype=np.uint8))
        with self.assertRaises(ImageNetInputError):
            normalize_crop(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float32))
        with self.assertRaises(ImageNetInputError):
            quantize_input(np.zeros((1, 3, 8, 8), dtype=np.float32), self.scale)
        with self.assertRaises(ImageNetInputError):
            quantize_input(normalize_crop(self.crop), 0.0)
        with self.assertRaises(ImageNetInputError):
            dequantized_preview(
                np.zeros((1, CROP_SIZE, CROP_SIZE, 3), dtype=np.int16), self.scale
            )


class DecodeTest(unittest.TestCase):
    def setUp(self):
        self.names = tuple(f"class-{index}" for index in range(1000))

    def test_top_k_is_ordered_and_normalized(self):
        logits = np.full((1, 1000), -5, dtype=np.int8)
        logits[0, 7] = 100
        logits[0, 11] = 60
        logits[0, 3] = 20
        predictions = decode_logits(logits, 0.1, self.names, top_k=3)
        self.assertEqual([item.index for item in predictions], [7, 11, 3])
        self.assertEqual(predictions[0].name, "class-7")
        self.assertAlmostEqual(predictions[0].logit, 10.0, places=9)
        self.assertGreater(predictions[0].probability, predictions[1].probability)
        self.assertTrue(
            all(0.0 <= item.probability <= 1.0 for item in predictions)
        )

    def test_full_probability_distribution_sums_to_one(self):
        rng = np.random.default_rng(7)
        logits = rng.integers(-127, 128, size=(1, 1000)).astype(np.int8)
        predictions = decode_logits(logits, 0.05, self.names, top_k=1000)
        total = sum(item.probability for item in predictions)
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_record_is_json_ready(self):
        logits = np.zeros((1, 1000), dtype=np.int8)
        record = decode_logits(logits, 0.1, self.names, top_k=1)[0].as_record()
        self.assertEqual(
            sorted(record), ["index", "logit", "name", "probability"]
        )
        self.assertIsInstance(record["index"], int)
        self.assertIsInstance(record["name"], str)

    def test_malformed_decode_arguments_are_rejected(self):
        logits = np.zeros((1, 1000), dtype=np.int8)
        with self.assertRaises(ImageNetInputError):
            decode_logits(logits.astype(np.int32), 0.1, self.names)
        with self.assertRaises(ImageNetInputError):
            decode_logits(np.zeros((1, 10), dtype=np.int8), 0.1, self.names)
        with self.assertRaises(ImageNetInputError):
            decode_logits(logits, 0.0, self.names)
        with self.assertRaises(ImageNetInputError):
            decode_logits(logits, 0.1, self.names, top_k=0)


class ClassNameTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "imagenet-classes.txt"

    def write(self, count: int) -> None:
        self.path.write_text(
            "\n".join(f"class-{index}" for index in range(count)) + "\n",
            encoding="utf-8",
        )

    def test_exactly_one_thousand_names_are_required(self):
        self.write(1000)
        names = load_class_names(self.path)
        self.assertEqual(len(names), 1000)
        self.assertEqual(names[0], "class-0")
        self.write(999)
        with self.assertRaises(ImageNetInputError):
            load_class_names(self.path)

    def test_index_or_exact_name_resolves_the_expected_class(self):
        self.write(1000)
        names = load_class_names(self.path)
        self.assertEqual(resolve_class_index("258", names), 258)
        self.assertEqual(resolve_class_index("class-258", names), 258)
        for selector in ("", "1000", "no-such-class"):
            with self.assertRaises(ImageNetInputError):
                resolve_class_index(selector, names)


if __name__ == "__main__":
    unittest.main()
