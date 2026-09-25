"""The single execution path.

Every surface funnels through `dispatch`. The CLI in-process, the CLI against
a daemon, the HTTP endpoint, the MCP tool and the pi extension all end up
here, which is what makes their results identical rather than merely similar.

`dispatch` is responsible for three things and nothing else: validate the
input against the category's model, open a run directory when the op records
one, and hand the backend a context.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from . import paths
from .errors import IcError, InvalidInput
from .registry import Backend, Category, Op, find_op, resolve_backend
from .runstore import Run, RunStore


@dataclass
class Ctx:
    """What a backend is given. Backends never reach outside this."""

    run: Run
    cwd: Path
    backend_version: str
    store: RunStore
    parent_run: str | None = None

    def resolve(self, handle: str) -> Path:
        """Turn a `run_id/filename` handle into a path on disk."""
        return self.store.resolve(handle)


class _NullRun:
    """Stand-in for ops that do not record a run.

    A waveform query is read-only and happens dozens of times per debugging
    session; giving each one a run directory would bury the sim runs that
    matter under noise.
    """

    def __init__(self, scratch: Path):
        self.run_id = None
        self.dir = scratch
        self.artifacts = scratch
        self.work = scratch
        scratch.mkdir(parents=True, exist_ok=True)

        self.final_dir = scratch

    def handle(self, name: str) -> str:
        return str(self.dir / name)

    def final_artifact(self, name: str) -> Path:
        return self.dir / name


def _files_of(params: BaseModel) -> list[str] | None:
    value = getattr(params, "files", None)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return None


def validate(op: Op, payload: dict) -> BaseModel:
    try:
        return op.In.model_validate(payload)
    except ValidationError as exc:
        raise InvalidInput(
            f"invalid input for {op.name!r}",
            errors=[
                {"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]}
                for e in exc.errors()
            ],
        ) from exc


def dispatch(op_name: str, payload: dict, *, cwd: Path | None = None,
             store: RunStore | None = None) -> dict:
    category, op = find_op(op_name)
    params = validate(op, payload)
    entry = resolve_backend(category, getattr(params, "backend", None))
    # The caller's directory, so relative source paths mean what the caller
    # typed. The store's location is a separate question, answered by paths.
    cwd = Path(cwd) if cwd else Path.cwd()
    store = store or RunStore()
    return _execute(category, op, entry, params, cwd, store)


def _execute(category: Category, op: Op, entry: Backend, params: BaseModel,
             cwd: Path, store: RunStore) -> dict:
    impl = entry.impl()
    method = getattr(impl, op.name, None)
    if method is None:
        raise IcError(
            f"backend {entry.name!r} does not implement {op.name!r}",
            category=category.name,
        )
    version = entry.version()
    inputs = params.model_dump(exclude_none=False)
    started = time.monotonic()

    if not op.records_run:
        scratch = store.root.parent / "scratch"
        ctx = Ctx(_NullRun(scratch), cwd, version, store)  # type: ignore[arg-type]
        result = method(params, ctx)
        return result.model_dump()

    with store.begin(
        category=category.name,
        op=op.name,
        backend=entry.name,
        backend_version=version,
        inputs={k: v for k, v in inputs.items() if k != "files"},
        input_files=_files_of(params),
    ) as run:
        ctx = Ctx(run, cwd, version, store)
        try:
            result = method(params, ctx)
        except IcError:
            raise
        except Exception as exc:  # backend crashed: still leave a run behind
            run.finish(
                state="failed",
                out={"error": str(exc)},
                duration_s=time.monotonic() - started,
                top=getattr(params, "top", None),
            )
            raise
        payload = result.model_dump()
        payload["run_id"] = run.run_id
        run.finish(
            state="succeeded" if payload.get("ok", True) else "failed",
            out=payload,
            exit_code=payload.get("exit_code"),
            duration_s=time.monotonic() - started,
            top=getattr(params, "top", None),
            parent_run=ctx.parent_run,
        )
        return payload
