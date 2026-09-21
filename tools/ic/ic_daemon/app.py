"""The daemon.

It does two things: persist run records and manage job lifecycle. It holds no
tool knowledge whatsoever -- every endpoint below is generated from the
registry, so a new category appears here the moment its folder exists.

FastAPI takes the categories' pydantic models directly, which is why the HTTP
schema, the MCP schema and the CLI arguments cannot drift: they are the same
objects.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from ic_core import paths
from ic_core.dispatch import dispatch, validate
from ic_core.errors import IcError
from ic_core.registry import iter_ops
from ic_core.runstore import RunStore
from ic_core.tools import import_errors

from .jobs import JobRunner, JobStore

#: Default long-poll window for a job. Comfortably inside the tool timeout of
#: every client tested, and long enough that a 40-minute synthesis costs about
#: eight calls rather than eighty.
DEFAULT_WAIT_S = 240.0

#: Artifact reads are for peeking at a log, not for shipping it to an agent.
ARTIFACT_LINE_CAP = 500


def create_app(root: Path | None = None, max_concurrent: int = 1) -> FastAPI:
    project_root = root or Path.cwd()
    store = RunStore()
    jobs = JobStore()
    runner = JobRunner(jobs, max_concurrent=max_concurrent)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Reconcile jobs left `running` by a daemon that died, so no client
        # waits forever on work nothing is doing.
        jobs.recover()
        yield

    app = FastAPI(
        title="ic-toold",
        version="0.1.0",
        description="Localhost daemon for the ic design tools. Run records and job lifecycle only.",
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.jobs = jobs
    app.state.runner = runner
    app.state.root = project_root

    @app.exception_handler(IcError)
    async def _ic_error(_request: Request, exc: IcError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    router = APIRouter(prefix="/v1")

    @router.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "root": str(project_root),
            "store": str(store.root),
            "ops": [op.name for _, op in iter_ops()],
        }

    @router.get("/tools")
    def tools() -> dict:
        """The tool catalogue every client renders from."""
        return {"tools": [describe(category, op) for category, op in iter_ops()]}

    @router.get("/backends")
    def backends() -> dict:
        from ic_core.registry import CATEGORIES, load_all

        load_all()
        out = []
        for category in CATEGORIES.values():
            for name, entry in category.backends.items():
                out.append(
                    {
                        "category": category.name,
                        "backend": name,
                        "default": name == category.default_backend,
                        "requires": entry.requires,
                        "available": entry.available(),
                        "version": entry.version() if entry.available() else None,
                    }
                )
        return {"backends": out, "import_errors": import_errors()}

    # One POST endpoint per op, all generated. Adding a category adds its
    # endpoints without this file changing.
    for category, op in iter_ops():
        _mount(router, category, op, runner, project_root)

    @router.get("/jobs")
    def list_jobs(limit: int = Query(20, ge=1, le=200)) -> dict:
        return {"jobs": jobs.list(limit)}

    @router.get("/jobs/{job_id}")
    async def job_status(
        job_id: str, wait_s: float = Query(0.0, ge=0.0, le=3600.0)
    ) -> dict:
        if wait_s > 0:
            return await runner.wait(job_id, wait_s)
        record = jobs.get(job_id)
        return record or {"job_id": job_id, "state": "unknown"}

    @router.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        return runner.cancel(job_id)

    @router.get("/runs")
    def list_runs(
        category: str | None = None,
        top: str | None = None,
        state: str | None = None,
        limit: int = Query(20, ge=1, le=200),
    ) -> dict:
        return {"runs": store.list_runs(category=category, top=top, state=state, limit=limit)}

    @router.get("/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        return {"meta": store.meta(run_id), "out": store.out(run_id)}

    @router.get("/artifacts/{run_id}/{name}")
    def artifact(
        run_id: str,
        name: str,
        tail: int = Query(100, ge=0, le=ARTIFACT_LINE_CAP),
        head: int = Query(0, ge=0, le=ARTIFACT_LINE_CAP),
        grep: str | None = None,
    ) -> dict:
        """Peek at an artifact without moving it.

        Bounded on purpose: a 12 MB simulation log must stay on disk. What an
        agent needs is the last hundred lines or the lines matching a pattern.
        """
        path = store.resolve(f"{run_id}/{name}")
        size = path.stat().st_size
        lines = path.read_text(errors="replace").splitlines()
        total = len(lines)
        if grep:
            needle = grep.lower()
            lines = [ln for ln in lines if needle in ln.lower()][:ARTIFACT_LINE_CAP]
        elif head:
            lines = lines[:head]
        else:
            lines = lines[-tail:] if tail else []
        return {
            "handle": f"{run_id}/{name}",
            "path": str(path),
            "bytes": size,
            "total_lines": total,
            "lines": lines,
            "truncated": len(lines) < total,
        }

    app.include_router(router)
    return app


def describe(category, op) -> dict:
    return {
        "name": op.name,
        "category": category.name,
        "summary": op.summary,
        "long_running": op.long_running,
        "records_run": op.records_run,
        "default_backend": category.default_backend,
        "backends": sorted(category.backends),
        "input_schema": op.In.model_json_schema(),
        "output_schema": op.Out.model_json_schema(),
    }


def _mount(router: APIRouter, category, op, runner: JobRunner, root: Path) -> None:
    InModel = op.In

    if op.long_running:

        async def endpoint(params, wait_s: float = DEFAULT_WAIT_S):
            """Submit as a job, then long poll for `wait_s`.

            A short run answers inline and the caller never learns a job
            existed. A long one comes back as `{job_id, state: "running"}`,
            to be picked up with GET /v1/jobs/{job_id}?wait_s=...
            """
            job_id = runner.submit(op.name, params.model_dump(), cwd=root)
            if wait_s <= 0:
                return {"job_id": job_id, "state": "queued", "op": op.name}
            record = await runner.wait(job_id, wait_s)
            if record.get("result"):
                return {**record["result"], "job_id": job_id}
            return {
                "job_id": job_id,
                "state": record.get("state", "running"),
                "op": op.name,
                "still_running": record.get("still_running", True),
                "error": record.get("error"),
                "hint": f"GET /v1/jobs/{job_id}?wait_s=240 to keep waiting",
            }

        annotations = {
            "params": Annotated[InModel, Body(...)],
            "wait_s": Annotated[float, Query(ge=0.0, le=3600.0)],
            "return": Any,
        }
    else:

        def endpoint(params):
            return dispatch(op.name, params.model_dump(), cwd=root)

        annotations = {"params": Annotated[InModel, Body(...)], "return": Any}

    # This module uses `from __future__ import annotations`, so every
    # annotation written above is a string that FastAPI cannot resolve back to
    # a category's model -- the model is a closure variable, not a module
    # attribute. Attaching the real objects is what lets one generic mount
    # function serve every category.
    endpoint.__annotations__ = annotations
    endpoint.__name__ = f"op_{op.name}"

    router.add_api_route(
        f"/tools/{op.name}",
        endpoint,
        methods=["POST"],
        summary=op.summary,
        response_model=None,
        tags=[category.name],
    )


__all__ = ["create_app", "describe", "validate"]
