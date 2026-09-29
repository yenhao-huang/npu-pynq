"""Run persistence.

One tool invocation produces one directory:

    .ic/runs/2026-09-21/143052-a3f21c-sim/
        meta.json      what ran, whether it worked, how long it took
        out.json       the summary the agent received
        artifacts/     wave.fst, sim.log, utilization.rpt
        work/          Verilator obj_dir, Vivado project

The three levels have different lifetimes: `meta.json` and `out.json` are kept
forever (kilobytes), `artifacts/` has a retention policy (megabytes to
gigabytes), and `work/` is reclaimed first. Deleting `artifacts/` leaves the
history intact.

Grouping by run rather than by tool is deliberate. The question asked after the
fact is almost always "what did that last run do" -- and a single sim produces
both a waveform and a log, which are one event. Splitting by tool would file
two halves of one answer in two places.

`meta.json` is the truth. `index.db` only holds the columns worth querying and
can be deleted and rebuilt from the tree at any time.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import paths
from .errors import HandleNotFound

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    category    TEXT NOT NULL,
    op          TEXT NOT NULL,
    backend     TEXT,
    top         TEXT,
    state       TEXT NOT NULL,
    duration_s  REAL,
    dir         TEXT NOT NULL,
    parent_run  TEXT
);
CREATE INDEX IF NOT EXISTS runs_ts ON runs (ts DESC);
CREATE INDEX IF NOT EXISTS runs_top ON runs (top, category, ts DESC);
"""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


class Run:
    """A run directory being written. Not valid until `finish` renames it."""

    def __init__(self, store: "RunStore", run_id: str, tmp_dir: Path, final_dir: Path,
                 category: str, op: str, backend: str | None):
        self.store = store
        self.run_id = run_id
        self.dir = tmp_dir
        self.final_dir = final_dir
        self.category = category
        self.op = op
        self.backend = backend
        self.meta: dict = {}
        self.dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)

    @property
    def artifacts(self) -> Path:
        return self.dir / "artifacts"

    @property
    def work(self) -> Path:
        return self.dir / "work"

    def final_artifact(self, artifact_name: str) -> Path:
        """Where an artifact will live once the run is committed.

        `self.dir` still carries the `.tmp` suffix while a backend is writing,
        so anything that reports a path a person will later type -- a GTKWave
        save file, say -- must use this rather than the working path.
        """
        return self.final_dir / "artifacts" / artifact_name

    def handle(self, artifact_name: str) -> str:
        """A handle is `run_id/filename` -- a direct map to a path on disk.

        It stays valid across daemon restarts because nothing has to be
        rebuilt to interpret it.
        """
        return f"{self.run_id}/{artifact_name}"

    def finish(self, *, state: str, out: dict, exit_code: int | None = None,
               duration_s: float | None = None, top: str | None = None,
               parent_run: str | None = None) -> Path:
        artifacts = []
        if self.artifacts.exists():
            for item in sorted(self.artifacts.iterdir()):
                if item.is_file():
                    artifacts.append({"name": item.name, "bytes": item.stat().st_size})
        self.meta.update(
            {
                "run_id": self.run_id,
                "category": self.category,
                "op": self.op,
                "backend": self.backend,
                "state": state,
                "exit_code": exit_code,
                "duration_s": round(duration_s, 3) if duration_s is not None else None,
                "artifacts": artifacts,
                "parent_run": parent_run,
                "finished_at": _now().isoformat(),
            }
        )
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2) + "\n")
        (self.dir / "out.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
        # Rename last: a directory that still carries the `.tmp` suffix was
        # interrupted, and GC can remove it without ever mistaking a partial
        # result for a real one.
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.dir, self.final_dir)
        self.dir = self.final_dir
        self.store._index(self.meta, self.final_dir, top)
        return self.final_dir


class RunStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else paths.runs_root()
        self.db_path = (self.root.parent / "index.db")

    # -- writing ---------------------------------------------------------

    @contextmanager
    def begin(self, *, category: str, op: str, backend: str | None,
              inputs: dict, backend_version: str | None = None,
              input_files: list[str] | None = None):
        run_id = uuid.uuid4().hex[:6]
        now = _now()
        day = self.root / now.strftime("%Y-%m-%d")
        name = f"{now.strftime('%H%M%S')}-{run_id}-{category}"
        run = Run(self, run_id, day / (name + ".tmp"), day / name, category, op, backend)
        recorded = dict(inputs)
        if input_files is not None:
            recorded["files"] = [
                {"path": str(f), "sha256": sha256_of(Path(f))}
                if Path(f).is_file()
                else {"path": str(f), "sha256": None}
                for f in input_files
            ]
        run.meta = {
            "run_id": run_id,
            "category": category,
            "op": op,
            "backend": backend,
            "backend_version": backend_version,
            "inputs": recorded,
            "started_at": now.isoformat(),
            "state": "running",
        }
        (run.dir / "meta.json").write_text(json.dumps(run.meta, indent=2) + "\n")
        try:
            yield run
        except Exception:
            if run.dir.exists() and run.dir.name.endswith(".tmp"):
                run.meta["state"] = "failed"
                (run.dir / "meta.json").write_text(json.dumps(run.meta, indent=2) + "\n")
            raise

    # -- index -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.executescript(SCHEMA)
        return conn

    def _index(self, meta: dict, directory: Path, top: str | None) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, ts, category, op, backend, top, state, duration_s, dir, parent_run) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    meta["run_id"],
                    meta.get("started_at", ""),
                    meta.get("category", ""),
                    meta.get("op", ""),
                    meta.get("backend"),
                    top or (meta.get("inputs") or {}).get("top"),
                    meta.get("state", "unknown"),
                    meta.get("duration_s"),
                    str(directory),
                    meta.get("parent_run"),
                ),
            )

    def rebuild_index(self) -> int:
        """`index.db` is disposable; this reconstructs it from meta.json files."""
        if self.db_path.exists():
            self.db_path.unlink()
        count = 0
        for meta_file in sorted(self.root.glob("*/*/meta.json")):
            if meta_file.parent.name.endswith(".tmp"):
                continue
            meta = json.loads(meta_file.read_text())
            self._index(meta, meta_file.parent, (meta.get("inputs") or {}).get("top"))
            count += 1
        return count

    # -- reading ---------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT dir FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row and Path(row[0]).exists():
            return Path(row[0])
        # The index can lag or be absent; the tree is the truth.
        for candidate in self.root.glob(f"*/*-{run_id}-*"):
            if candidate.is_dir() and not candidate.name.endswith(".tmp"):
                return candidate
        raise HandleNotFound(f"no run {run_id!r} in {self.root}", run_id=run_id)

    def meta(self, run_id: str) -> dict:
        return json.loads((self.run_dir(run_id) / "meta.json").read_text())

    def out(self, run_id: str) -> dict:
        path = self.run_dir(run_id) / "out.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def resolve(self, handle: str) -> Path:
        """Turn a `run_id/filename` handle back into a path.

        A bare path is passed through, so a human can hand the tools a waveform
        that was never produced by a run.
        """
        if "/" not in handle:
            direct = Path(handle)
            if direct.exists():
                return direct
            raise HandleNotFound(f"malformed handle {handle!r}; expected 'run_id/filename'")
        run_id, _, name = handle.partition("/")
        if len(run_id) != 6 or not all(c in "0123456789abcdef" for c in run_id):
            direct = Path(handle)
            if direct.exists():
                return direct
            raise HandleNotFound(f"no such file or handle: {handle!r}")
        path = self.run_dir(run_id) / "artifacts" / name
        if not path.exists():
            raise HandleNotFound(f"run {run_id!r} has no artifact {name!r}", handle=handle)
        return path

    def list_runs(self, *, category: str | None = None, top: str | None = None,
                  state: str | None = None, limit: int = 20) -> list[dict]:
        clauses, args = [], []
        for column, value in (("category", category), ("top", top), ("state", state)):
            if value:
                clauses.append(f"{column} = ?")
                args.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT run_id, ts, category, op, backend, top, state, duration_s, dir, parent_run "
                f"FROM runs {where} ORDER BY ts DESC LIMIT ?",
                (*args, limit),
            ).fetchall()
        keys = ["run_id", "ts", "category", "op", "backend", "top", "state",
                "duration_s", "dir", "parent_run"]
        return [dict(zip(keys, row)) for row in rows]

    # -- retention -------------------------------------------------------

    def gc(self, *, drop_work: bool = True, drop_artifacts_before: str | None = None,
           drop_tmp: bool = True) -> dict:
        """Reclaim space in the order the levels were designed to be reclaimed."""
        removed = {"tmp": 0, "work": 0, "artifacts": 0}
        for day in sorted(self.root.glob("*")):
            if not day.is_dir():
                continue
            for run in sorted(day.iterdir()):
                if not run.is_dir():
                    continue
                if run.name.endswith(".tmp"):
                    if drop_tmp:
                        shutil.rmtree(run, ignore_errors=True)
                        removed["tmp"] += 1
                    continue
                if drop_work and (run / "work").exists():
                    shutil.rmtree(run / "work", ignore_errors=True)
                    removed["work"] += 1
                if drop_artifacts_before and day.name < drop_artifacts_before:
                    if (run / "artifacts").exists():
                        shutil.rmtree(run / "artifacts", ignore_errors=True)
                        removed["artifacts"] += 1
        return removed
