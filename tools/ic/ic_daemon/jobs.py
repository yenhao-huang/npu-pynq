"""Job lifecycle -- about 150 lines, no framework.

    queued -> running -> succeeded
                      -> failed
                      -> cancelled

Celery and RQ both want a broker. Huey can use SQLite but still brings a worker
model and task registration. What is actually needed here is "spawn a thing and
remember it", for a single user, on one machine, with jobs numbered in the
single digits -- so a broker would be pure overhead.

State lives in SQLite rather than in memory so a client can ask about a job
after the daemon restarted, and so a crashed daemon leaves a recoverable
record instead of a lost one.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import closing
from pathlib import Path

from ic_core import paths
from ic_core.dispatch import dispatch

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    op          TEXT NOT NULL,
    state       TEXT NOT NULL,
    pid         INTEGER,
    created_at  REAL NOT NULL,
    started_at  REAL,
    ended_at    REAL,
    run_id      TEXT,
    result      TEXT,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs (state, created_at DESC);
"""

TERMINAL = {"succeeded", "failed", "cancelled"}


class JobStore:
    def __init__(self, db_path: Path | None = None):
        import sqlite3

        self.sqlite3 = sqlite3
        self.db_path = db_path or paths.jobs_db()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        return self.sqlite3.connect(self.db_path, timeout=30)

    def create(self, op: str) -> str:
        job_id = uuid.uuid4().hex[:8]
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO jobs (job_id, op, state, created_at) VALUES (?,?,?,?)",
                (job_id, op, "queued", time.time()),
            )
        return job_id

    def update(self, job_id: str, **fields) -> None:
        if not fields:
            return
        if "result" in fields and fields["result"] is not None:
            fields["result"] = json.dumps(fields["result"], default=str)
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id = ?",
                (*fields.values(), job_id),
            )

    def get(self, job_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            conn.row_factory = self.sqlite3.Row
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        if record.get("result"):
            record["result"] = json.loads(record["result"])
        return record

    def list(self, limit: int = 20) -> list[dict]:
        with closing(self._connect()) as conn:
            conn.row_factory = self.sqlite3.Row
            rows = conn.execute(
                "SELECT job_id, op, state, created_at, started_at, ended_at, run_id "
                "FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def recover(self) -> int:
        """Reconcile after a restart.

        A row that says `running` whose pid is gone belongs to a daemon that
        died. Marking it `failed` is honest; leaving it would strand a client
        waiting forever on a job nothing is working on.
        """
        recovered = 0
        with closing(self._connect()) as conn:
            conn.row_factory = self.sqlite3.Row
            rows = conn.execute(
                "SELECT job_id, pid FROM jobs WHERE state IN ('running','queued')"
            ).fetchall()
        for row in rows:
            pid = row["pid"]
            alive = False
            if pid:
                try:
                    os.kill(pid, 0)
                    alive = True
                except OSError:
                    alive = False
            if not alive:
                self.update(
                    row["job_id"],
                    state="failed",
                    ended_at=time.time(),
                    error="daemon restarted while this job was running",
                )
                recovered += 1
        return recovered


class JobRunner:
    """Runs long ops off the request path.

    The semaphore is not optional. A single Vivado instance is several
    gigabytes; two on a desktop means swapping, and a third means the machine
    stops responding.
    """

    def __init__(self, store: JobStore, max_concurrent: int = 1):
        self.store = store
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tasks: dict[str, asyncio.Task] = {}
        self.events: dict[str, asyncio.Event] = {}

    def submit(self, op: str, payload: dict, cwd: Path | None = None) -> str:
        job_id = self.store.create(op)
        self.events[job_id] = asyncio.Event()
        self.tasks[job_id] = asyncio.create_task(self._run(job_id, op, payload, cwd))
        return job_id

    async def _run(self, job_id: str, op: str, payload: dict, cwd: Path | None) -> None:
        async with self.semaphore:
            record = self.store.get(job_id)
            if record and record["state"] == "cancelled":
                self.events[job_id].set()
                return
            self.store.update(
                job_id, state="running", pid=os.getpid(), started_at=time.time()
            )
            try:
                # dispatch blocks on a subprocess; a thread keeps the event
                # loop free to answer status and cancel requests meanwhile.
                result = await asyncio.to_thread(dispatch, op, payload, cwd=cwd)
                self.store.update(
                    job_id,
                    state="succeeded" if result.get("ok", True) else "failed",
                    ended_at=time.time(),
                    run_id=result.get("run_id"),
                    result=result,
                )
            except asyncio.CancelledError:
                self.store.update(job_id, state="cancelled", ended_at=time.time())
                raise
            except Exception as exc:
                self.store.update(
                    job_id, state="failed", ended_at=time.time(), error=str(exc)
                )
            finally:
                self.events[job_id].set()

    async def wait(self, job_id: str, timeout: float) -> dict:
        """Long poll.

        A 40-minute synthesis answered by polling every 30 seconds leaves 80
        `{"status":"running"}` objects in the agent's context. Long polling
        turns the same wait into roughly 8 calls, each of which carried
        information.
        """
        event = self.events.get(job_id)
        record = self.store.get(job_id)
        if record is None:
            return {"job_id": job_id, "state": "unknown"}
        if record["state"] in TERMINAL or event is None:
            return record
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return {**self.store.get(job_id), "still_running": True}
        return self.store.get(job_id)

    def cancel(self, job_id: str) -> dict:
        task = self.tasks.get(job_id)
        record = self.store.get(job_id)
        if record is None:
            return {"job_id": job_id, "state": "unknown"}
        if record["state"] in TERMINAL:
            return record
        if task is not None:
            task.cancel()
        self.store.update(job_id, state="cancelled", ended_at=time.time())
        return self.store.get(job_id)
