"""Contract tests for release version selection and real-image acceptance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "resnet18"
RESOLVER = REPOSITORY_ROOT / ".github" / "cd" / "resolve_release_version.py"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise AssertionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def git(*arguments: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


class ReleaseVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "changelog").mkdir()
        git("init", "-q", "-b", "main", cwd=self.root)
        git("config", "user.email", "test@example.com", cwd=self.root)
        git("config", "user.name", "Test", cwd=self.root)
        self.resolver = load_module("resolve_release_version", RESOLVER)

    def _changelog(self, name: str, body: str = "notes\n") -> None:
        (self.root / "changelog" / name).write_text(body, encoding="utf-8")

    def _commit(self) -> None:
        git("add", "-A", cwd=self.root)
        git("commit", "-q", "-m", "release", cwd=self.root)

    def test_highest_declared_version_is_selected(self) -> None:
        for name in ("v0.1.3.md", "v0.1.10.md", "v0.1.9.md"):
            self._changelog(name)
        self._commit()
        resolution = self.resolver.resolve(self.root)
        self.assertEqual(resolution["release_tag"], "v0.1.10")
        self.assertTrue(resolution["should_release"])
        self.assertEqual(resolution["changelog_path"], "changelog/v0.1.10.md")
        self.assertEqual(resolution["release_commit"], git("rev-parse", "HEAD", cwd=self.root))

    def test_already_tagged_version_publishes_nothing(self) -> None:
        self._changelog("v0.2.0.md")
        self._commit()
        git("tag", "-a", "v0.2.0", "-m", "v0.2.0", cwd=self.root)
        resolution = self.resolver.resolve(self.root)
        self.assertFalse(resolution["should_release"])
        self.assertIn("already tagged", resolution["reason"])

    def test_empty_or_missing_declaration_is_rejected(self) -> None:
        self._changelog("v0.2.0.md", "   \n")
        self._commit()
        with self.assertRaisesRegex(self.resolver.ReleaseVersionError, "empty"):
            self.resolver.resolve(self.root)
        (self.root / "changelog" / "v0.2.0.md").unlink()
        self._commit()
        with self.assertRaisesRegex(self.resolver.ReleaseVersionError, "no version"):
            self.resolver.resolve(self.root)

    def test_unversioned_changelog_file_is_rejected(self) -> None:
        self._changelog("v0.2.0.md")
        self._changelog("draft.md")
        self._commit()
        with self.assertRaisesRegex(self.resolver.ReleaseVersionError, "not a version"):
            self.resolver.resolve(self.root)

    def test_cli_writes_workflow_outputs(self) -> None:
        self._changelog("v0.3.1.md")
        self._commit()
        output = self.root / "outputs.env"
        result = subprocess.run(
            [
                sys.executable,
                str(RESOLVER),
                "--repository-root",
                str(self.root),
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        written = dict(
            line.split("=", 1)
            for line in output.read_text(encoding="utf-8").splitlines()
        )
        self.assertEqual(written["release_tag"], "v0.3.1")
        self.assertEqual(written["should_release"], "true")
        self.assertEqual(json.loads(result.stdout)["release_tag"], "v0.3.1")


class ImageAcceptanceContractTests(unittest.TestCase):
    """The real-image acceptance is board-only and provenance-bound."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (EXAMPLE_ROOT / "accept_image_on_board.py").read_text(
            encoding="utf-8"
        )

    def test_only_the_physical_runtime_may_pass(self) -> None:
        self.assertIn("PASS [physical-pynq-z1]", self.source)
        self.assertIn("isinstance(physical, NPURuntime)", self.source)
        self.assertIn("result.metrics.physical_jobs <= 0", self.source)
        # A host backend must never be reachable from this entry point.
        self.assertNotIn("HostMatrixBackend", self.source)

    def test_evidence_binds_package_image_and_result(self) -> None:
        for marker in (
            '"release_tag"',
            '"source_commit"',
            '"bit_sha256"',
            '"hwh_sha256"',
            '"top_k"',
            '"verdict"',
            "NPU_RESNET18_IMAGE_ACCEPTANCE",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.source)

    def test_board_captures_are_compared_against_the_host_record(self) -> None:
        self.assertIn("board capture differs from host", self.source)
        self.assertIn("_array_sha256", self.source)

    def test_the_entry_point_ships_inside_the_release_package(self) -> None:
        packager = load_module(
            "resnet18_package_delivery", EXAMPLE_ROOT / "package_example.py"
        )
        self.assertIn(
            "accept_image_on_board.py",
            packager.DELIVERY_SOURCE_FILES.values(),
        )
        self.assertIn(
            "image-acceptance.json",
            (EXAMPLE_ROOT / "run_on_board.py").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
