"""Where things live.

Two separate questions that are easy to conflate:

- **Which directory do relative paths in a request resolve against?**
  The caller's working directory. A person in `src/hw/` typing
  `ic lint --files rtl/npu_pe.sv` means the file next to them.
- **Where does the run store live?**
  One place per project, so the CLI, the daemon, the MCP server and the pi
  extension all see the same handles no matter where each was started.
  `IC_ROOT` overrides it, which is what tests and CI use.

Conflating the two makes `IC_ROOT` silently relocate the working directory and
break every relative source path, so they stay apart here.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """The enclosing repository, found by walking up to the nearest `.git`."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
    return here


def store_root(start: Path | None = None) -> Path:
    env = os.environ.get("IC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return project_root(start) / ".ic"


def runs_root(start: Path | None = None) -> Path:
    return store_root(start) / "runs"


def index_db(start: Path | None = None) -> Path:
    return store_root(start) / "index.db"


def jobs_db(start: Path | None = None) -> Path:
    return store_root(start) / "jobs.db"
