from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest

import numpy as np

from src.export.resnet import export_model
from src.model.package import REQUIRED_ABI_MAJOR, REQUIRED_CAPABILITIES
from src.model.resnet18 import load_acceptance_bundle
from src.runtime.acceptance import AcceptanceRunError, run_resnet18_acceptance
from src.runtime.model import NPUModelRuntime, load_model_package
from src.test.tests.resnet18_fixture import (
    make_reduced_resnet18_graph,
    write_canonical_json,
    write_deterministic_npz,
)


class MatrixFake:
    abi_major = REQUIRED_ABI_MAJOR
    capabilities = REQUIRED_CAPABILITIES
    max_m = 2
    max_n = 2
    max_k = 2

    def __init__(self, *, cycles=3):
        self.cycles = cycles
        self.fail_next = False
        self.calls = 0

    def run(self, a, b, **_timeouts):
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise TimeoutError("injected acceptance failure")
        if self.cycles is not None:
            self.last_metrics = SimpleNamespace(cycles=self.cycles)
        elif hasattr(self, "last_metrics"):
            del self.last_metrics
        return np.asarray(a, dtype=np.int32) @ np.asarray(b, dtype=np.int32)


class TickingClock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1_000_000
        return self.value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResNet18RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.graph = make_reduced_resnet18_graph("runner")
        package = export_model(self.graph, self.root / "model")
        self.model = load_model_package(package.manifest_path)
        self.capture_name = self.graph.commands[3].output_id
        inputs = np.stack(
            (
                np.full((1, 8, 8, 1), -2, dtype=np.int8),
                np.full((1, 8, 8, 1), 3, dtype=np.int8),
            )
        )
        reference_runtime = NPUModelRuntime(MatrixFake(), self.model)
        reference_results = [
            reference_runtime.run(
                {self.graph.inputs[0]: source},
                capture_tensors=(self.capture_name,),
            )
            for source in inputs
        ]
        outputs = np.stack(
            [result.outputs[self.graph.outputs[0]] for result in reference_results]
        )
        captures = np.stack(
            [result.captures[self.capture_name] for result in reference_results]
        )
        labels = np.argmax(outputs.reshape(2, -1), axis=1).astype(np.int64)
        corpus_path = self.root / "corpus.npz"
        write_deterministic_npz(
            corpus_path,
            {
                "capture_0": captures,
                "expected_outputs": outputs,
                "inputs": inputs,
                "labels": labels,
                "sample_ids": np.array(["sample-a", "sample-b"]),
            },
        )
        descriptor_path = self.root / "acceptance.json"
        descriptor = {
            "assets": {
                "corpus": self.asset(corpus_path),
                "model_manifest": self.asset(package.manifest_path),
                "model_payload": self.asset(package.payload_path),
            },
            "capture_tensors": [self.capture_name],
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
                "require_cycles": True,
                "top1_min": 1.0,
            },
        }
        write_canonical_json(descriptor_path, descriptor)
        self.bundle = load_acceptance_bundle(descriptor_path, graph=self.graph)

    @staticmethod
    def asset(path):
        return {
            "bytes": path.stat().st_size,
            "filename": path.name,
            "sha256": digest(path),
        }

    def test_complete_fixture_acceptance_and_atomic_evidence(self):
        physical = MatrixFake()
        runtime = NPUModelRuntime(physical, self.model)

        def recovery_probe():
            physical.fail_next = True
            runtime.run({self.graph.inputs[0]: self.bundle.corpus.inputs[0]})

        evidence_path = self.root / "evidence.json"
        evidence = run_resnet18_acceptance(
            self.bundle,
            runtime,
            evidence_path=evidence_path,
            recovery_probe=recovery_probe,
            monotonic_ns=TickingClock(),
        )
        self.assertEqual(evidence["evidence_type"], "software-integration")
        self.assertEqual(evidence["sample_ids"], ["sample-a", "sample-b"])
        self.assertEqual(evidence["performance"]["invocations"], 4)
        self.assertTrue(evidence["gates"]["recovery_injected"])
        self.assertGreater(evidence["performance"]["physical_jobs"], 0)
        self.assertGreater(evidence["performance"]["physical_cycles"], 0)
        self.assertEqual(
            evidence_path.read_bytes().decode("utf-8").count("\n"), 1
        )
        with self.assertRaises(TypeError):
            evidence["mode"] = "board"

    def test_mismatch_and_missing_cycles_preserve_prior_evidence(self):
        evidence_path = self.root / "evidence.json"
        evidence_path.write_bytes(b'{"known":"good"}\n')
        previous = evidence_path.read_bytes()
        corrupted = self.bundle.corpus.expected_outputs.copy()
        corrupted[0] ^= np.int8(1)
        corrupted.flags.writeable = False
        corpus = replace(self.bundle.corpus, expected_outputs=corrupted)
        bundle = replace(self.bundle, corpus=corpus)
        with self.assertRaisesRegex(AcceptanceRunError, "exact output ratio"):
            run_resnet18_acceptance(
                bundle,
                NPUModelRuntime(MatrixFake(), self.model),
                evidence_path=evidence_path,
                monotonic_ns=TickingClock(),
            )
        self.assertEqual(evidence_path.read_bytes(), previous)

        with self.assertRaisesRegex(AcceptanceRunError, "cycle telemetry"):
            run_resnet18_acceptance(
                self.bundle,
                NPUModelRuntime(MatrixFake(cycles=None), self.model),
                evidence_path=evidence_path,
                monotonic_ns=TickingClock(),
            )
        self.assertEqual(evidence_path.read_bytes(), previous)

    def test_capture_mismatch_identifies_sample_and_tensor(self):
        changed = self.bundle.corpus.expected_captures[self.capture_name].copy()
        changed[0].flat[0] ^= np.int8(1)
        changed.flags.writeable = False
        corpus = replace(
            self.bundle.corpus,
            expected_captures=MappingProxyType({self.capture_name: changed}),
        )
        with self.assertRaisesRegex(
            AcceptanceRunError, "sample-a.*runner_block0_conv1"
        ):
            run_resnet18_acceptance(
                replace(self.bundle, corpus=corpus),
                NPUModelRuntime(MatrixFake(), self.model),
                monotonic_ns=TickingClock(),
            )


if __name__ == "__main__":
    unittest.main()
