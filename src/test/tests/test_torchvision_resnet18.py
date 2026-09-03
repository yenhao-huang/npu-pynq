from __future__ import annotations

import unittest

import numpy as np

from src.export.torchvision_resnet18 import (
    FIXTURE_EVIDENCE_TYPE,
    PHYSICAL_BOARD_EVIDENCE_TYPE,
    REAL_MODEL_HOST_EVIDENCE_TYPE,
    activation_quantization,
    compare_integer_captures,
    encode_q31_ratio,
    expected_source_shapes,
    fold_batch_norm,
    generate_calibration_inputs,
    quantize_conv_weight,
    residual_scale_groups,
    validate_source_arrays,
    _safe_quantize_conv,
)


class TorchVisionResNet18ConversionTests(unittest.TestCase):
    def test_fixed_source_schema_rejects_missing_extra_and_wrong_shape(self):
        shapes = expected_source_shapes()
        self.assertEqual(shapes["conv1.weight"], (64, 3, 7, 7))
        self.assertEqual(shapes["fc.weight"], (1000, 512))
        self.assertEqual(shapes["layer4.0.downsample.0.weight"], (512, 256, 1, 1))
        state = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in shapes.items()
        }
        validate_source_arrays(state)
        missing = dict(state)
        del missing["conv1.weight"]
        with self.assertRaisesRegex(ValueError, "conv1.weight"):
            validate_source_arrays(missing)
        extra = dict(state)
        extra["unexpected.weight"] = np.zeros((1,), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "unexpected.weight"):
            validate_source_arrays(extra)
        wrong = dict(state)
        wrong["fc.weight"] = np.zeros((999, 512), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "fc.weight"):
            validate_source_arrays(wrong)
        nonfinite = dict(state)
        nonfinite["fc.bias"] = nonfinite["fc.bias"].copy()
        nonfinite["fc.bias"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_source_arrays(nonfinite)

    def test_batch_norm_fold_and_hwio_weight_conversion(self):
        weight = np.array(
            [
                [[[1.0, -2.0], [3.0, -4.0]]],
                [[[-0.5, 1.0], [-1.5, 2.0]]],
            ],
            dtype=np.float32,
        )
        bias = np.array([0.25, -0.5], dtype=np.float32)
        gamma = np.array([2.0, -3.0], dtype=np.float32)
        beta = np.array([0.1, 0.2], dtype=np.float32)
        mean = np.array([0.5, -0.25], dtype=np.float32)
        variance = np.array([4.0, 9.0], dtype=np.float32)
        folded_weight, folded_bias = fold_batch_norm(
            weight, bias, gamma, beta, mean, variance, epsilon=0.0
        )
        np.testing.assert_allclose(
            folded_weight,
            weight * np.array([1.0, -1.0], dtype=np.float32)[:, None, None, None],
        )
        np.testing.assert_allclose(
            folded_bias,
            np.array([-0.15, 0.45], dtype=np.float32),
        )
        quantized, scales = quantize_conv_weight(folded_weight)
        self.assertEqual(quantized.shape, (2, 2, 1, 2))
        self.assertEqual(quantized.dtype, np.int8)
        self.assertEqual(scales.shape, (2,))

    def test_near_dead_batch_norm_channel_remains_accumulator_safe(self):
        weight = np.full((1, 3, 7, 7), 1e-16, dtype=np.float32)
        bias = np.array([1.0], dtype=np.float32)
        quantized, scales = _safe_quantize_conv(weight, bias, 0.02)
        quantized_bias = int(np.floor(abs(bias[0] / (0.02 * scales[0])) + 0.5))
        bound = quantized_bias + 128 * int(
            np.sum(np.abs(quantized.astype(np.int16)), dtype=np.int64)
        )
        self.assertLessEqual(bound, (1 << 31) - 1)

    def test_activation_and_q31_parameters_are_signed_and_deterministic(self):
        quantization = activation_quantization(2.54)
        self.assertEqual(quantization.zero_point, 0)
        self.assertGreater(quantization.multiplier_q31, 0)
        self.assertEqual(quantization.shift, 0)
        first = encode_q31_ratio(0.125)
        second = encode_q31_ratio(0.125)
        self.assertEqual(first, second)
        self.assertEqual(first, (1 << 28, 0))
        with self.assertRaises(ValueError):
            encode_q31_ratio(1.0)

    def test_residual_scale_groups_preserve_every_add_contract(self):
        observed = {
            "stem.relu": 3.0,
            "stem.pool": 2.5,
            "layer1.0.conv2": 4.0,
            "layer1.0.add": 4.5,
            "layer1.0.relu": 4.5,
            "layer1.1.conv2": 3.5,
            "layer1.1.add": 5.0,
            "layer1.1.relu": 5.0,
            "layer2.0.conv2": 6.0,
            "layer2.0.projection": 5.5,
            "layer2.0.add": 7.0,
            "layer2.0.relu": 7.0,
            "layer2.1.conv2": 6.5,
            "layer2.1.add": 8.0,
            "layer2.1.relu": 8.0,
        }
        scales = residual_scale_groups(observed, stages=2)
        stage_one = {
            scales[name]
            for name in observed
            if name.startswith(("stem", "layer1"))
        }
        stage_two = {
            scales[name]
            for name in observed
            if name.startswith("layer2") and not name.endswith("conv1")
        }
        self.assertEqual(len(stage_one), 1)
        self.assertEqual(len(stage_two), 1)
        self.assertGreater(next(iter(stage_two)), next(iter(stage_one)))

    def test_full_shape_calibration_and_evidence_levels(self):
        inputs = generate_calibration_inputs()
        self.assertEqual(inputs.dtype, np.float32)
        self.assertEqual(inputs.shape[1:], (3, 224, 224))
        self.assertGreaterEqual(inputs.shape[0], 2)
        self.assertEqual(
            len(
                {
                    FIXTURE_EVIDENCE_TYPE,
                    REAL_MODEL_HOST_EVIDENCE_TYPE,
                    PHYSICAL_BOARD_EVIDENCE_TYPE,
                }
            ),
            3,
        )
        self.assertNotIn("board", REAL_MODEL_HOST_EVIDENCE_TYPE)

    def test_first_stage_capture_mismatch_is_explicit(self):
        expected = {
            "stem.relu": np.zeros((1, 112, 112, 64), dtype=np.int8),
            "layer1.1.relu": np.zeros((1, 56, 56, 64), dtype=np.int8),
            "logits": np.zeros((1, 1000), dtype=np.int8),
        }
        actual = {name: value.copy() for name, value in expected.items()}
        actual["layer1.1.relu"][0, 2, 3, 4] = 1
        with self.assertRaisesRegex(
            ValueError, r"layer1.1.relu.*\[0, 2, 3, 4\]"
        ):
            compare_integer_captures(expected, actual)


if __name__ == "__main__":
    unittest.main()
