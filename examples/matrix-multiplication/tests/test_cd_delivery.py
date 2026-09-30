from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "matrix-multiplication"
SKILL_ROOT = (
    REPOSITORY_ROOT / ".codex" / "skills" / "deploy" / "release-npu-pynq"
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from src.runtime.verify_overlay import write_manifest
from src.test.tests.test_verify_overlay import HWH


def load_required_module(name: str, path: Path):
    if not path.is_file():
        raise AssertionError(f"required module is missing: {path}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise AssertionError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def create_artifacts(directory: Path, source_commit: str = "a" * 40) -> None:
    directory.mkdir(parents=True)
    (directory / "npu_matrix.bit").write_bytes(b"release-bitstream")
    (directory / "npu_matrix.hwh").write_text(HWH, encoding="utf-8")
    write_manifest(
        directory,
        source_commit=source_commit,
        vivado_version="2026.1",
    )


class FakePhysicalRuntime:
    max_m = 2
    max_n = 2
    max_k = 256

    def run(
        self,
        a_matrix: np.ndarray,
        b_matrix: np.ndarray,
        *,
        hardware_timeout_cycles: int,
        software_timeout: float,
    ) -> np.ndarray:
        del hardware_timeout_cycles, software_timeout
        return np.asarray(a_matrix, dtype=np.int32) @ np.asarray(
            b_matrix, dtype=np.int32
        )


class CorruptPhysicalRuntime(FakePhysicalRuntime):
    def run(self, *args, **kwargs) -> np.ndarray:  # type: ignore[no-untyped-def]
        result = super().run(*args, **kwargs)
        result[0, 0] += 1
        return result


class StandalonePackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.artifact_dir = self.root / "artifacts"
        self.output_dir = self.root / "package"
        create_artifacts(self.artifact_dir)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_package_uses_explicit_standalone_layout(self) -> None:
        package_module = load_required_module(
            "matrix_package_example",
            EXAMPLE_ROOT / "package_example.py",
        )

        manifest = package_module.build_package(
            repository_root=REPOSITORY_ROOT,
            artifact_dir=self.artifact_dir,
            output_dir=self.output_dir,
            release_tag="v0.1.1",
            source_commit="a" * 40,
        )

        expected_files = {
            "README.md",
            "matrix_multiplication.ipynb",
            "run_on_board.py",
            "runtime/matrix_multiplication.py",
            "src/runtime/__init__.py",
            "src/runtime/npu.py",
            "src/runtime/verify_overlay.py",
            "artifacts/npu_matrix.bit",
            "artifacts/npu_matrix.hwh",
            "artifacts/npu_matrix.manifest.json",
            "package.manifest.json",
        }
        actual_files = {
            path.relative_to(self.output_dir).as_posix()
            for path in self.output_dir.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        imported = subprocess.run(
            [sys.executable, "-c",
             "from src.runtime import load_pynq_runtime; "
             "from src.runtime.verify_overlay import verify_artifacts; "
             "from runtime.matrix_multiplication import TiledMatrixMultiplier"],
            cwd=self.output_dir, env=environment, capture_output=True, text=True,
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)
        self.assertEqual(manifest["release_tag"], "v0.1.1")
        self.assertEqual(manifest["source_commit"], "a" * 40)
        self.assertEqual(
            json.loads(
                (self.output_dir / "package.manifest.json").read_text(
                    encoding="utf-8"
                )
            ),
            manifest,
        )

    def test_missing_artifact_fails_without_output(self) -> None:
        package_module = load_required_module(
            "matrix_package_example_missing",
            EXAMPLE_ROOT / "package_example.py",
        )
        (self.artifact_dir / "npu_matrix.hwh").unlink()

        with self.assertRaises(Exception):
            package_module.build_package(
                repository_root=REPOSITORY_ROOT,
                artifact_dir=self.artifact_dir,
                output_dir=self.output_dir,
                release_tag="v0.1.1",
                source_commit="a" * 40,
            )

        self.assertFalse(self.output_dir.exists())

    def test_cli_imports_from_repo_and_infers_local_metadata(self) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)

        result = subprocess.run(
            [
                sys.executable,
                str(EXAMPLE_ROOT / "package_example.py"),
                "--artifact-dir",
                str(self.artifact_dir),
                "--output-dir",
                str(self.output_dir),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(
            (self.output_dir / "package.manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["release_tag"], "local-aaaaaaaa")
        self.assertEqual(manifest["source_commit"], "a" * 40)

        readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "python examples/matrix-multiplication/package_example.py\n",
            readme,
        )
        self.assertNotIn("--repository-root", readme)
        self.assertNotIn("--source-commit", readme)


class BoardRunnerTests(unittest.TestCase):
    def test_8x8_cases_cover_full_array_and_edge_tiles(self) -> None:
        board_module = load_required_module(
            "matrix_run_on_board_8x8", EXAMPLE_ROOT / "run_on_board.py"
        )
        runtime = FakePhysicalRuntime()
        runtime.max_m = runtime.max_n = 8
        evidence = board_module.execute_cases(runtime, {}, release_tag="v0.0.0")
        self.assertEqual([case["tile_count"] for case in evidence["cases"]], [1, 4, 4])
        self.assertEqual(evidence["cases"][0]["shape"][0], 8)
        self.assertEqual(evidence["cases"][1]["shape"][0], 9)

    def test_required_cases_and_evidence(self) -> None:
        board_module = load_required_module(
            "matrix_run_on_board",
            EXAMPLE_ROOT / "run_on_board.py",
        )
        manifest = {
            "source_commit": "a" * 40,
            "vivado_version": "2026.1",
            "target_part": "xc7z020clg400-1",
            "bit": {"sha256": "b" * 64},
            "hwh": {"sha256": "c" * 64},
        }

        evidence = board_module.execute_cases(
            FakePhysicalRuntime(),
            manifest,
            release_tag="v0.1.1",
        )

        self.assertEqual(evidence["release_tag"], "v0.1.1")
        self.assertEqual(evidence["source_commit"], "a" * 40)
        self.assertEqual(evidence["physical_limits"], [2, 2, 256])
        self.assertEqual(
            [case["name"] for case in evidence["cases"]],
            ["normal", "non_aligned", "repeated"],
        )
        self.assertEqual(evidence["cases"][1]["tile_count"], 4)
        self.assertTrue(
            all(case["status"] == "PASS" for case in evidence["cases"])
        )
        self.assertEqual(
            evidence["pass_marker"],
            "PASS: Phase 1C matrix multiplication example",
        )
        serialized = json.dumps(evidence, allow_nan=False, sort_keys=True)
        for forbidden in ("password", "private_key", "environment"):
            self.assertNotIn(forbidden, serialized.lower())

    def test_mismatch_does_not_produce_pass_evidence(self) -> None:
        board_module = load_required_module(
            "matrix_run_on_board_failure",
            EXAMPLE_ROOT / "run_on_board.py",
        )

        with self.assertRaises(board_module.BoardExampleError):
            board_module.execute_cases(
                CorruptPhysicalRuntime(),
                {"source_commit": "a" * 40},
                release_tag="v0.1.1",
            )


class DeploymentWrapperTests(unittest.TestCase):
    def test_dry_run_validates_package_without_network_commands(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        self.assertIsNotNone(powershell, "PowerShell is required for this test")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_dir = root / "artifacts"
            package_dir = root / "package"
            create_artifacts(artifact_dir)
            package_module = load_required_module(
                "matrix_package_for_deploy",
                EXAMPLE_ROOT / "package_example.py",
            )
            package_module.build_package(
                repository_root=REPOSITORY_ROOT,
                artifact_dir=artifact_dir,
                output_dir=package_dir,
                release_tag="v0.1.1",
                source_commit="a" * 40,
            )

            result = subprocess.run(
                [
                    str(powershell),
                    "-NoProfile",
                    "-File",
                    str(EXAMPLE_ROOT / "deploy_release.ps1"),
                    "-PackagePath",
                    str(package_dir),
                    "-ReleaseTag",
                    "v0.1.1",
                    "-DeploymentId",
                    "test-1",
                    "-EvidencePath",
                    str(root / "board-evidence.json"),
                    "-DryRun",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("no archive or network command was executed", result.stdout)
            self.assertFalse((root / "board-evidence.json").exists())

        script = (EXAMPLE_ROOT / "deploy_release.ps1").read_text(encoding="utf-8")
        for forbidden in ("password", "private_key", "sshpass"):
            self.assertNotIn(forbidden, script.lower())

    def test_deploy_sources_xrt_and_pynq_venv_before_board_execution(self) -> None:
        script = (EXAMPLE_ROOT / "deploy_release.ps1").read_text(encoding="utf-8")
        xrt_path = "/etc/profile.d/xrt_setup.sh"
        venv_path = "/etc/profile.d/pynq_venv.sh"
        xrt_source_command = f"source {xrt_path}"
        source_command = f"source {venv_path}"
        board_command = "python3 run_on_board.py"

        self.assertIn(f"test -r {xrt_path}", script)
        self.assertIn(xrt_source_command, script)
        self.assertIn(f"test -r {venv_path}", script)
        self.assertIn(source_command, script)
        self.assertIn(board_command, script)
        self.assertLess(script.index(xrt_source_command), script.index(source_command))
        self.assertLess(script.index(source_command), script.index(board_command))

    def test_deploy_uses_root_pynq_for_mmio_and_supports_local_prompt(self) -> None:
        script = (EXAMPLE_ROOT / "deploy_release.ps1").read_text(encoding="utf-8")

        self.assertIn("[switch]$InteractiveSudo", script)
        self.assertIn("sudo ${sudoArguments}XILINX_XRT=/usr", script)
        self.assertIn("/usr/local/share/pynq-venv/bin/python3", script)
        self.assertIn("$validationSshArguments += '-tt'", script)


class ReleaseWorkflowContractTests(unittest.TestCase):
    """A release branch is validated first; only then can a Release exist."""

    def setUp(self) -> None:
        workflows = REPOSITORY_ROOT / ".github" / "workflows"
        self.cd_path = workflows / "cd.yml"
        self.publish_path = workflows / "release-publish.yml"
        self.assertTrue(self.cd_path.is_file(), "cd.yml is required")
        self.assertTrue(self.publish_path.is_file(), "release-publish.yml is required")
        self.cd = self.cd_path.read_text(encoding="utf-8")
        self.publish = self.publish_path.read_text(encoding="utf-8")

    def test_cd_runs_on_a_release_branch_before_main(self) -> None:
        import yaml

        cd = yaml.safe_load(self.cd)
        triggers = cd[True] if True in cd else cd["on"]
        self.assertIn("release/v*", triggers["push"]["branches"])
        # A push event uses the workflow from the branch it ran on, which is what
        # lets CD validate itself before anything reaches main.
        self.assertNotIn("branches: [main]", self.cd)
        self.assertNotIn("types: [published]", self.cd)
        self.assertFalse(
            (REPOSITORY_ROOT / ".github" / "workflows" / "build.yml").exists(),
            "tag-push build workflow must be retired",
        )

    def test_the_branch_name_must_match_the_declared_version(self) -> None:
        self.assertIn("resolve_release_version.py --format tag", self.cd)
        self.assertIn("but the changelog declares", self.cd)
        self.assertIn("is already tagged", self.cd)

    def test_privileged_work_stays_on_trusted_runners(self) -> None:
        for marker in (
            "runs-on: [self-hosted, vivado]",
            "runs-on: [self-hosted, pynq-z1]",
            "environment: pynq-z1-production",
            "build/vivado/npu_matrix_8x8/artifacts",
            "build/vivado/npu_matrix_8x8/reports/build_evidence.txt",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.cd)

    def test_a_passing_run_produces_a_draft_and_nothing_public(self) -> None:
        import yaml

        cd = yaml.safe_load(self.cd)
        draft = cd["jobs"]["publish-draft"]
        self.assertEqual(
            set(draft["needs"]),
            {"validate-source", "host-checks", "build-overlay", "board-validation"},
        )
        self.assertEqual(draft["permissions"]["contents"], "write")
        self.assertIn("--draft", self.cd)
        # CD must never create the tag; that belongs to the publish workflow.
        self.assertNotIn("git tag", self.cd)
        self.assertNotIn("--draft=false", self.cd)
        for name, job in cd["jobs"].items():
            if name == "publish-draft":
                continue
            with self.subTest(job=name):
                self.assertNotEqual(
                    job.get("permissions", {}).get("contents"), "write"
                )

    def test_the_draft_carries_packages_evidence_and_checksums(self) -> None:
        for marker in (
            "npu-resnet18-",
            "npu-matrix-",
            "SHA256SUMS",
            "sha256sum --check --strict",
            "resnet18-board-evidence.json",
            "resnet18-image-acceptance.json",
            "board-evidence.json",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.cd)

    def test_publishing_requires_a_validated_and_merged_draft(self) -> None:
        import yaml

        publish = yaml.safe_load(self.publish)
        triggers = publish[True] if True in publish else publish["on"]
        self.assertEqual(set(triggers), {"workflow_dispatch"})
        job = publish["jobs"]["publish"]
        self.assertEqual(job["environment"], "pynq-z1-release")
        self.assertEqual(job["permissions"]["contents"], "write")
        for marker in (
            "isDraft",
            "is not a draft",
            "merge-base --is-ancestor",
            "not a squash",
            "is already tagged",
            "--draft=false",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.publish)
        # Publishing must not rebuild, repackage, or touch the board.
        for forbidden in ("vivado", "self-hosted", "package_example.py"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.publish)

    def test_pre_merge_gate_protects_main(self) -> None:
        ci = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        for marker in (
            "release-readiness:",
            "github.base_ref == 'main'",
            "resolve_release_version.py",
            "make -C src/test model",
            "release-publish.yml",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, ci)

    def test_the_release_skill_documents_the_whole_path(self) -> None:
        skill = SKILL_ROOT / "SKILL.md"
        self.assertTrue(skill.is_file(), "the release skill is required")
        body = skill.read_text(encoding="utf-8")
        for marker in (
            "release/vX.Y.Z",
            "draft Release",
            "merge commit",
            "release-publish.yml",
            "resolve_release_version.py",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, body)

    def test_the_release_skill_starts_the_runners_itself(self) -> None:
        body = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        # Bringing the runners online is the skill's job, and it must happen
        # before the push that starts the privileged run.
        self.assertIn("start_cd_runners.ps1", body)
        runners = body.index("Bring the runners online")
        branch = body.index("Cut the release branch")
        self.assertLess(runners, branch, "runners come online before the push")
        for marker in ("vivado", "pynq-z1", "queued"):
            with self.subTest(marker=marker):
                self.assertIn(marker, body)

        script = SKILL_ROOT / "references" / "scripts" / "start_cd_runners.ps1"
        self.assertTrue(script.is_file(), "the runner start script is required")
        source = script.read_text(encoding="utf-8")
        # Discover services; a hardcoded runner name would not survive a rename.
        self.assertIn("actions.runner.*", source)
        self.assertIn("Start-Service", source)
        self.assertIn("SupportsShouldProcess", source)
        # A runner may be a service or an interactive run.cmd; handle both.
        self.assertIn("run.cmd", source)
        self.assertIn("Start-Process", source)
        # Starting a service is the whole remit: no registration, no tokens.
        for forbidden in ("config.cmd", "--token", "svc.cmd install", "Remove-Service"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

        for reference in ("references/rules/env.md", "references/runners.md"):
            with self.subTest(reference=reference):
                self.assertTrue((SKILL_ROOT / reference).is_file(), reference)


if __name__ == "__main__":
    unittest.main()
