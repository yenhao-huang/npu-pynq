from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.export.resnet import export_model
from src.model.resnet import ConstantTensor, Conv2D, QuantizedGraph
from src.model.resnet18 import (
    AcceptanceValidationError,
    load_acceptance_bundle,
    validate_resnet18_topology,
)
from src.test.tests.resnet18_fixture import (
    make_reduced_resnet18_graph,
    write_canonical_json,
    write_deterministic_npz,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AcceptanceBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.graph = make_reduced_resnet18_graph("opaque")
        package = export_model(self.graph, self.root / "model")
        self.corpus_path = self.root / "corpus.npz"
        write_deterministic_npz(
            self.corpus_path,
            {
                "expected_outputs": np.zeros((2, 1, 2), dtype=np.int8),
                "inputs": np.zeros((2, 1, 8, 8, 1), dtype=np.int8),
                "labels": np.array([0, 1], dtype=np.int64),
                "sample_ids": np.array(["a", "b"], dtype="<U1"),
            },
        )
        self.descriptor_path = self.root / "acceptance.json"
        self.descriptor = {
            "assets": {
                "corpus": self.asset(self.corpus_path),
                "model_manifest": self.asset(package.manifest_path),
                "model_payload": self.asset(package.payload_path),
            },
            "capture_tensors": [],
            "class_count": 2,
            "format": {"major": 1, "minor": 0},
            "magic": "NPU_RESNET18_ACCEPTANCE",
            "reference": {
                "framework": "fixture",
                "model_id": "reduced-resnet18",
                "preprocessing_id": "signed-int8-v1",
                "version": "1",
            },
            "sample_count": 2,
            "thresholds": {
                "exact_output_min": 1.0,
                "require_cycles": False,
                "top1_min": 0.0,
            },
        }
        write_canonical_json(self.descriptor_path, self.descriptor)

    @staticmethod
    def asset(path):
        return {
            "bytes": path.stat().st_size,
            "filename": path.name,
            "sha256": digest(path),
        }

    def test_valid_bundle_and_name_independent_topology(self):
        bundle = load_acceptance_bundle(
            self.descriptor_path, graph=self.graph
        )
        self.assertEqual(bundle.corpus.inputs.shape, (2, 1, 8, 8, 1))
        self.assertFalse(bundle.corpus.inputs.flags.writeable)
        topology = validate_resnet18_topology(self.graph)
        renamed = validate_resnet18_topology(
            make_reduced_resnet18_graph("nothing_semantic")
        )
        self.assertEqual(len(topology.blocks), 8)
        self.assertEqual(topology.projection_blocks, (2, 4, 6))
        self.assertEqual(len(renamed.blocks), 8)

    def test_digest_path_and_noncanonical_descriptor_fail_closed(self):
        self.corpus_path.write_bytes(self.corpus_path.read_bytes() + b"x")
        with self.assertRaisesRegex(AcceptanceValidationError, "digest|length"):
            load_acceptance_bundle(self.descriptor_path)

        self.descriptor["assets"]["corpus"]["filename"] = "../corpus.npz"
        write_canonical_json(self.descriptor_path, self.descriptor)
        with self.assertRaisesRegex(AcceptanceValidationError, "basename"):
            load_acceptance_bundle(self.descriptor_path)

        self.descriptor["assets"]["corpus"] = self.asset(self.corpus_path)
        self.descriptor_path.write_text(
            json.dumps(self.descriptor, indent=2), encoding="utf-8"
        )
        with self.assertRaisesRegex(AcceptanceValidationError, "canonical"):
            load_acceptance_bundle(self.descriptor_path)

    def test_object_corpus_and_inconsistent_samples_are_rejected(self):
        np.savez(
            self.corpus_path,
            inputs=np.zeros((2, 1, 8, 8, 1), dtype=np.int8),
            expected_outputs=np.zeros((2, 1, 2), dtype=np.int8),
            labels=np.array([0, 1], dtype=np.int64),
            sample_ids=np.array([object(), object()], dtype=object),
        )
        self.descriptor["assets"]["corpus"] = self.asset(self.corpus_path)
        write_canonical_json(self.descriptor_path, self.descriptor)
        with self.assertRaisesRegex(AcceptanceValidationError, "pickle|object"):
            load_acceptance_bundle(self.descriptor_path)

        write_deterministic_npz(
            self.corpus_path,
            {
                "expected_outputs": np.zeros((1, 1, 2), dtype=np.int8),
                "inputs": np.zeros((2, 1, 8, 8, 1), dtype=np.int8),
                "labels": np.array([0, 1], dtype=np.int64),
                "sample_ids": np.array(["a", "b"], dtype="<U1"),
            },
        )
        self.descriptor["assets"]["corpus"] = self.asset(self.corpus_path)
        write_canonical_json(self.descriptor_path, self.descriptor)
        with self.assertRaisesRegex(AcceptanceValidationError, "sample"):
            load_acceptance_bundle(self.descriptor_path)

    def test_noncanonical_projection_is_rejected(self):
        projection = next(
            command
            for command in self.graph.commands
            if isinstance(command, Conv2D)
            and "block2_projection_cmd" in command.command_id
        )
        old_weight = next(
            item for item in self.graph.constants
            if item.name == projection.weight_id
        )
        replacement = ConstantTensor(
            old_weight.name,
            (3, 3, old_weight.shape[2], old_weight.shape[3]),
            "int8",
            "HWIO",
            (1,) * (9 * old_weight.shape[2] * old_weight.shape[3]),
        )
        constants = tuple(
            replacement if item.name == old_weight.name else item
            for item in self.graph.constants
        )
        commands = tuple(
            replace(command, padding=(1, 1, 1, 1))
            if command.command_id == projection.command_id else command
            for command in self.graph.commands
        )
        malformed = QuantizedGraph(
            self.graph.tensors,
            constants,
            commands,
            self.graph.inputs,
            self.graph.outputs,
        )
        with self.assertRaisesRegex(
            AcceptanceValidationError, "projection.*1x1"
        ):
            validate_resnet18_topology(malformed)


if __name__ == "__main__":
    unittest.main()
