from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.export.resnet import (
    ExportError,
    certify_accumulators,
    export_model,
)
from src.model.numeric import INT32_MAX
from src.model.package import PackageValidationError, validate_package_data
from src.model.package import (
    REQUIRED_ABI_MAJOR as PACKAGE_ABI_MAJOR,
    REQUIRED_CAPABILITIES as PACKAGE_CAPABILITIES,
)
from src.model.resnet import (
    ConstantTensor,
    Conv2D,
    Flatten,
    FullyConnected,
    GlobalAveragePool,
    Quantization,
    QuantizedGraph,
    TensorSpec,
)
from src.runtime.npu import REQUIRED_CAPABILITIES as RUNTIME_CAPABILITIES
from src.runtime.npu import VERSION_MAJOR as RUNTIME_ABI_MAJOR


Q = Quantization(INT32_MAX, 0, 0)


def activation(name, shape, layout="NHWC"):
    return TensorSpec(name, shape, layout, Q)


def graph_fixture(*, conv_weight=2, conv_bias=7):
    tensors = (
        activation("input", (1, 2, 2, 1)),
        activation("conv_out", (1, 2, 2, 1)),
        activation("avg_out", (1, 1, 1, 1)),
        activation("flat_out", (1, 1), "NC"),
        activation("logits", (1, 2), "NC"),
    )
    constants = (
        ConstantTensor(
            "z_fc_bias", (2,), "int32", "BIAS", (11, -13)
        ),
        ConstantTensor(
            "a_conv_weight", (1, 1, 1, 1), "int8", "HWIO", (conv_weight,)
        ),
        ConstantTensor(
            "m_fc_weight", (1, 2), "int8", "IO", (3, -4)
        ),
        ConstantTensor(
            "b_conv_bias", (1,), "int32", "BIAS", (conv_bias,)
        ),
    )
    commands = (
        Conv2D(
            "conv",
            "input",
            "a_conv_weight",
            "conv_out",
            (INT32_MAX,),
            (0,),
            bias_id="b_conv_bias",
        ),
        GlobalAveragePool("average", "conv_out", "avg_out"),
        Flatten("flatten", "avg_out", "flat_out"),
        FullyConnected(
            "fc",
            "flat_out",
            "m_fc_weight",
            "logits",
            (INT32_MAX, INT32_MAX),
            (0, 0),
            bias_id="z_fc_bias",
        ),
    )
    return QuantizedGraph(
        tensors=tensors,
        constants=constants,
        commands=commands,
        inputs=("input",),
        outputs=("logits",),
    )


class AccumulatorCertificateTests(unittest.TestCase):
    def test_exact_boundary_is_accepted(self):
        graph = graph_fixture(conv_weight=0, conv_bias=INT32_MAX)
        certificates = certify_accumulators(graph)
        convolution = next(item for item in certificates if item.command_id == "conv")
        self.assertEqual(convolution.bounds, (INT32_MAX,))

    def test_unsafe_channel_is_rejected(self):
        graph = graph_fixture(conv_weight=1, conv_bias=INT32_MAX)
        with self.assertRaisesRegex(ExportError, "conv.*channel 0"):
            certify_accumulators(graph)

    def test_fully_connected_bounds_are_reported_per_channel(self):
        certificates = certify_accumulators(graph_fixture())
        fully_connected = next(
            item for item in certificates if item.command_id == "fc"
        )
        self.assertEqual(fully_connected.bounds, (11 + 128 * 3, 13 + 128 * 4))


class DeterministicExportTests(unittest.TestCase):
    def test_package_requirements_match_phase1_runtime(self):
        self.assertEqual(PACKAGE_ABI_MAJOR, RUNTIME_ABI_MAJOR)
        self.assertEqual(PACKAGE_CAPABILITIES, RUNTIME_CAPABILITIES)

    def test_repeated_exports_are_byte_identical_and_path_free(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            graph = graph_fixture()
            reordered = replace(
                graph,
                tensors=tuple(reversed(graph.tensors)),
                constants=tuple(reversed(graph.constants)),
            )
            first = export_model(graph, Path(first_dir) / "resnet")
            second = export_model(reordered, Path(second_dir) / "resnet")
            first_manifest = first.manifest_path.read_bytes()
            second_manifest = second.manifest_path.read_bytes()
            first_payload = first.payload_path.read_bytes()
            second_payload = second.payload_path.read_bytes()
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_payload, second_payload)
            self.assertEqual(first.payload_sha256, second.payload_sha256)
            manifest_text = first_manifest.decode("utf-8")
            self.assertNotIn(first_dir, manifest_text)
            self.assertNotIn(second_dir, manifest_text)
            self.assertNotIn("timestamp", manifest_text.lower())
            self.assertNotIn("created", manifest_text.lower())

            manifest = json.loads(manifest_text)
            constant_names = [entry["name"] for entry in manifest["constants"]]
            self.assertEqual(constant_names, sorted(constant_names))
            offsets = [entry["offset"] for entry in manifest["constants"]]
            self.assertEqual(offsets, sorted(offsets))
            for offset in offsets:
                self.assertEqual(offset % 64, 0)
            validate_package_data(manifest, first_payload)

    def test_corrupt_payload_and_overlapping_ranges_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            exported = export_model(graph_fixture(), Path(directory) / "resnet")
            manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
            payload = bytearray(exported.payload_path.read_bytes())
            payload[0] ^= 0xFF
            with self.assertRaisesRegex(PackageValidationError, "digest"):
                validate_package_data(manifest, bytes(payload))

            manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
            manifest["constants"][1]["offset"] = manifest["constants"][0]["offset"]
            with self.assertRaisesRegex(PackageValidationError, "overlap"):
                validate_package_data(manifest, exported.payload_path.read_bytes())

            manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
            manifest["accumulator_certificates"][0]["bounds"][0] = INT32_MAX + 1
            with self.assertRaisesRegex(PackageValidationError, "bound"):
                validate_package_data(manifest, exported.payload_path.read_bytes())

    def test_validation_failure_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "unsafe"
            with self.assertRaises(ExportError):
                export_model(
                    graph_fixture(conv_weight=1, conv_bias=INT32_MAX),
                    prefix,
                )
            self.assertFalse(prefix.with_suffix(".npu.json").exists())
            self.assertFalse(prefix.with_suffix(".npu.bin").exists())

    def test_publish_failure_restores_existing_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "resnet"
            original = export_model(graph_fixture(conv_weight=2), prefix)
            original_manifest = original.manifest_path.read_bytes()
            original_payload = original.payload_path.read_bytes()
            real_replace = os.replace
            calls = 0

            def fail_manifest_once(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected manifest replace failure")
                return real_replace(source, destination)

            with patch("src.export.resnet.os.replace", side_effect=fail_manifest_once):
                with self.assertRaisesRegex(ExportError, "publish"):
                    export_model(graph_fixture(conv_weight=3), prefix)

            self.assertEqual(original_manifest, original.manifest_path.read_bytes())
            self.assertEqual(original_payload, original.payload_path.read_bytes())
            leftovers = list(Path(directory).glob(".*.tmp"))
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
