"""Select the release version one validated `main` commit should publish.

The version is a human decision recorded in `changelog/vMAJOR.MINOR.PATCH.md`
on `main`, not a workflow input. This script reads that decision, refuses an
ambiguous or already-published version, and reports whether this commit has a
release to publish. It performs no network access and creates no tag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHANGELOG_PATTERN = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)\.md$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ReleaseVersionError(RuntimeError):
    """The release version cannot be resolved from the repository state."""


def _git(*arguments: str, repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleaseVersionError(f"git {arguments[0]} failed: {error}") from error
    return completed.stdout.strip()


def changelog_versions(changelog_dir: Path) -> list[tuple[tuple[int, int, int], Path]]:
    """Return every released version the changelog directory declares."""

    if not changelog_dir.is_dir():
        raise ReleaseVersionError("changelog directory is missing")
    versions: list[tuple[tuple[int, int, int], Path]] = []
    for path in sorted(changelog_dir.iterdir()):
        if not path.is_file():
            continue
        match = CHANGELOG_PATTERN.match(path.name)
        if match is None:
            raise ReleaseVersionError(f"changelog file is not a version: {path.name}")
        if not path.read_text(encoding="utf-8").strip():
            raise ReleaseVersionError(f"changelog file is empty: {path.name}")
        versions.append(
            (
                (
                    int(match.group("major")),
                    int(match.group("minor")),
                    int(match.group("patch")),
                ),
                path,
            )
        )
    if not versions:
        raise ReleaseVersionError("changelog declares no version")
    return sorted(versions)


def existing_tags(repository_root: Path) -> set[str]:
    """Return the release tags this checkout already knows about."""

    output = _git("tag", "--list", "v*.*.*", repository_root=repository_root)
    return {line.strip() for line in output.splitlines() if line.strip()}


def resolve(repository_root: Path) -> dict[str, object]:
    """Decide the tag this commit publishes, or report that it publishes none."""

    versions = changelog_versions(repository_root / "changelog")
    version, changelog_path = versions[-1]
    tag = "v{}.{}.{}".format(*version)
    commit = _git("rev-parse", "HEAD", repository_root=repository_root).lower()
    if COMMIT_PATTERN.match(commit) is None:
        raise ReleaseVersionError("HEAD did not resolve to a full commit id")

    tags = existing_tags(repository_root)
    published = tag in tags
    return {
        "changelog_path": changelog_path.relative_to(repository_root).as_posix(),
        "release_commit": commit,
        "release_tag": tag,
        "reason": (
            f"{tag} is already tagged; this commit publishes no new release"
            if published
            else f"{tag} is declared by {changelog_path.name} and is not yet tagged"
        ),
        "should_release": not published,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--format",
        choices=("json", "tag"),
        default="json",
        help="print the whole resolution, or only the declared release tag",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="append key=value lines here, for example $GITHUB_OUTPUT",
    )
    arguments = parser.parse_args()
    try:
        resolution = resolve(arguments.repository_root.resolve())
    except ReleaseVersionError as error:
        print(f"release version resolution failed: {error}", file=sys.stderr)
        return 1
    if arguments.format == "tag":
        print(resolution["release_tag"])
    else:
        print(json.dumps(resolution, indent=2, sort_keys=True))
    if arguments.output is not None:
        with arguments.output.open("a", encoding="utf-8") as stream:
            for key in ("release_tag", "release_commit", "changelog_path"):
                stream.write(f"{key}={resolution[key]}\n")
            stream.write(
                "should_release={}\n".format(
                    "true" if resolution["should_release"] else "false"
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
