"""The `ic` command line.

Subcommands are not written out here. They are reflected from the categories'
pydantic models, so `ic lint --files ... --top ...` exists because `LintIn`
has those fields -- and a new category's subcommand appears the moment its
folder does.

The CLI serves two callers. A person gets readable JSON at a shell. An agent
without MCP -- which is every pi-agent setup, since pi has no built-in MCP
support -- gets the same tools by running the same commands, which is why the
CLI is a first-class surface rather than a debugging convenience.

With `IC_DAEMON_URL` set, every call goes over HTTP to the daemon instead of
running in-process. The results are identical because both paths end in
`ic_core.dispatch`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from ic_core import paths
from ic_core.errors import IcError
from ic_core.registry import CATEGORIES, iter_ops, load_all
from ic_core.runstore import RunStore

DEFAULT_WAIT_S = 240.0


# -- reflecting a pydantic model into argparse ---------------------------


#: `int | None` has origin `types.UnionType`, `Optional[int]` has origin
#: `typing.Union`. Both appear in the category models, so both are matched --
#: and neither is matched against `None`, which every plain type would equal.
_UNION_ORIGINS = tuple(o for o in (Union, getattr(types, "UnionType", None)) if o is not None)


def _unwrap(annotation):
    """Strip Optional/Union down to the type a flag should parse."""
    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS:
        options = [a for a in get_args(annotation) if a is not type(None)]
        if len(options) == 1:
            return _unwrap(options[0])
        # Genuinely mixed unions (e.g. int | float) parse as strings and are
        # coerced by pydantic on the far side, which is the authority anyway.
        return str
    return annotation


def add_model_arguments(parser: argparse.ArgumentParser, model) -> None:
    for name, field in model.model_fields.items():
        flag = "--" + name.replace("_", "-")
        annotation = _unwrap(field.annotation)
        origin = get_origin(annotation)
        help_text = field.description or ""

        if annotation is bool:
            # Paired flags rather than `--trace true`: `--no-trace` is what a
            # person types and what a shell script reads back cleanly.
            group = parser.add_mutually_exclusive_group()
            group.add_argument(flag, dest=name, action="store_true", default=None,
                               help=help_text)
            group.add_argument("--no-" + name.replace("_", "-"), dest=name,
                               action="store_false", default=None,
                               help=argparse.SUPPRESS)
        elif origin is list:
            parser.add_argument(flag, dest=name, nargs="*", default=None, help=help_text)
        elif origin is Literal:
            parser.add_argument(flag, dest=name, choices=list(get_args(annotation)),
                                default=None, help=help_text)
        elif annotation is int:
            parser.add_argument(flag, dest=name, type=int, default=None, help=help_text)
        elif annotation is float:
            parser.add_argument(flag, dest=name, type=float, default=None, help=help_text)
        else:
            parser.add_argument(flag, dest=name, default=None, help=help_text)


def payload_from(args: argparse.Namespace, model) -> dict:
    """Only fields the user actually gave; the model supplies the rest."""
    return {
        name: getattr(args, name)
        for name in model.model_fields
        if getattr(args, name, None) is not None
    }


# -- the two execution paths ---------------------------------------------


def call(op_name: str, payload: dict, *, daemon: str | None, wait_s: float,
         cwd: Path) -> dict:
    if daemon:
        return _over_http(op_name, payload, daemon, wait_s)
    from ic_core.dispatch import dispatch

    return dispatch(op_name, payload, cwd=cwd)


def _over_http(op_name: str, payload: dict, base: str, wait_s: float) -> dict:
    import urllib.error
    import urllib.request

    url = f"{base.rstrip('/')}/v1/tools/{op_name}?wait_s={wait_s}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=wait_s + 60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"error": {"code": "http_error", "message": body, "status": exc.code}}


def _get(base: str, path: str) -> dict:
    import urllib.request

    with urllib.request.urlopen(f"{base.rstrip('/')}{path}", timeout=600) as response:
        return json.loads(response.read())


# -- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    load_all()
    parser = argparse.ArgumentParser(
        prog="ic",
        description="IC design tools: lint, simulate, debug waveforms, view, synthesize.",
        epilog="Run `ic tools` for the machine-readable catalogue, `ic doctor` to check the toolchain.",
    )
    # Global flags live on a parent parser so they work on either side of the
    # subcommand. `ic --compact lint ...` and `ic lint ... --compact` both read
    # naturally, and requiring one order would be a papercut on every call.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--daemon", default=os.environ.get("IC_DAEMON_URL"),
                        help="Route through ic-toold at this URL instead of running in-process")
    common.add_argument("--wait-s", type=float, default=DEFAULT_WAIT_S,
                        help="Long-poll window for long-running ops")
    common.add_argument("--compact", action="store_true",
                        help="One-line JSON instead of indented")
    for action in common._actions:
        parser._add_action(action)
    sub = parser.add_subparsers(dest="command", required=True)

    for category, op in iter_ops():
        op_parser = sub.add_parser(
            op.name,
            help=op.summary.split(".")[0] + ".",
            description=f"[{category.name}] {op.summary}",
            parents=[common],
        )
        add_model_arguments(op_parser, op.In)
        op_parser.set_defaults(_op=op)

    meta = {
        "tools": "List every tool with its JSON schema",
        "doctor": "Report which backends are installed and usable",
        "runs": "List recent runs",
        "run": "Show one run's meta.json and out.json",
        "artifact": "Print part of an artifact by handle",
        "jobs": "List jobs",
        "job": "Show or wait on one job",
        "gc": "Reclaim disk: work/ first, then old artifacts/",
        "reindex": "Rebuild index.db from the run tree",
        "serve": "Run the daemon in the foreground",
    }
    for name, help_text in meta.items():
        meta_parser = sub.add_parser(name, help=help_text, parents=[common])
        if name == "run":
            meta_parser.add_argument("run_id")
        elif name == "artifact":
            meta_parser.add_argument("handle", help="run_id/filename")
            meta_parser.add_argument("--tail", type=int, default=40)
            meta_parser.add_argument("--head", type=int, default=0)
            meta_parser.add_argument("--grep")
        elif name == "job":
            meta_parser.add_argument("job_id")
            # Not `--wait-s`: that is global and defaults to a long poll.
            # Asking for a job's status should not block by default.
            meta_parser.add_argument("--poll-s", type=float, default=0.0, dest="job_wait_s",
                                     help="Block up to this many seconds for the job to finish")
        elif name == "runs":
            meta_parser.add_argument("--category")
            meta_parser.add_argument("--top")
            meta_parser.add_argument("--state")
            meta_parser.add_argument("--limit", type=int, default=20)
        elif name == "jobs":
            meta_parser.add_argument("--limit", type=int, default=20)
        elif name == "gc":
            meta_parser.add_argument("--keep-work", action="store_true",
                                     help="Do not delete work/ directories")
            meta_parser.add_argument("--artifacts-before",
                                     help="Delete artifacts/ in run days before this YYYY-MM-DD")
        elif name == "serve":
            meta_parser.add_argument("--host", default="127.0.0.1")
            meta_parser.add_argument("--port", type=int, default=8731)
            meta_parser.add_argument("--max-concurrent", type=int, default=1)
    return parser


# -- meta commands --------------------------------------------------------


def cmd_tools() -> dict:
    from ic_daemon.app import describe

    return {"tools": [describe(c, o) for c, o in iter_ops()]}


def cmd_doctor() -> dict:
    from ic_core.tools import import_errors

    load_all()
    rows, missing = [], []
    for category in CATEGORIES.values():
        for name, entry in category.backends.items():
            ok = entry.available()
            rows.append({
                "category": category.name,
                "backend": name,
                "default": name == category.default_backend,
                "requires": entry.requires,
                "available": ok,
                "version": entry.version() if ok else None,
            })
            if not ok and name == category.default_backend:
                missing.append(f"{category.name}: default backend {name!r} needs {entry.requires!r}")
    return {
        "ops": [o.name for _, o in iter_ops()],
        "backends": rows,
        "blocking": missing,
        "import_errors": import_errors(),
        "store": str(paths.store_root()),
        "ok": not missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    indent = None if args.compact else 2
    root = Path.cwd()

    def emit(value: Any) -> int:
        print(json.dumps(value, indent=indent, default=str))
        if isinstance(value, dict):
            if "error" in value:
                return 2
            if value.get("ok") is False:
                return 1
        return 0

    try:
        op = getattr(args, "_op", None)
        if op is not None:
            payload = payload_from(args, op.In)
            return emit(call(op.name, payload, daemon=args.daemon,
                             wait_s=args.wait_s, cwd=root))

        command = args.command
        if command == "tools":
            return emit(cmd_tools())
        if command == "doctor":
            return emit(cmd_doctor())
        if command == "serve":
            from ic_daemon.__main__ import main as serve

            sys.argv = ["ic-toold", "--host", args.host, "--port", str(args.port),
                        "--max-concurrent", str(args.max_concurrent)]
            return serve()

        store = RunStore()
        if command == "runs":
            if args.daemon:
                query = f"?limit={args.limit}"
                for key in ("category", "top", "state"):
                    if getattr(args, key):
                        query += f"&{key}={getattr(args, key)}"
                return emit(_get(args.daemon, "/v1/runs" + query))
            return emit({"runs": store.list_runs(category=args.category, top=args.top,
                                                 state=args.state, limit=args.limit)})
        if command == "run":
            if args.daemon:
                return emit(_get(args.daemon, f"/v1/runs/{args.run_id}"))
            return emit({"meta": store.meta(args.run_id), "out": store.out(args.run_id)})
        if command == "artifact":
            path = store.resolve(args.handle)
            lines = path.read_text(errors="replace").splitlines()
            total = len(lines)
            if args.grep:
                needle = args.grep.lower()
                lines = [ln for ln in lines if needle in ln.lower()]
            elif args.head:
                lines = lines[: args.head]
            else:
                lines = lines[-args.tail:]
            return emit({"handle": args.handle, "path": str(path), "total_lines": total,
                         "lines": lines, "truncated": len(lines) < total})
        if command == "jobs":
            if args.daemon:
                return emit(_get(args.daemon, f"/v1/jobs?limit={args.limit}"))
            from ic_daemon.jobs import JobStore

            return emit({"jobs": JobStore().list(args.limit)})
        if command == "job":
            if args.daemon:
                return emit(_get(args.daemon, f"/v1/jobs/{args.job_id}?wait_s={args.job_wait_s}"))
            from ic_daemon.jobs import JobStore

            record = JobStore().get(args.job_id)
            return emit(record or {"job_id": args.job_id, "state": "unknown"})
        if command == "gc":
            return emit(store.gc(drop_work=not args.keep_work,
                                 drop_artifacts_before=args.artifacts_before))
        if command == "reindex":
            return emit({"indexed": store.rebuild_index()})
        parser.error(f"unhandled command {command!r}")
    except IcError as exc:
        print(json.dumps(exc.to_dict(), indent=indent, default=str), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
