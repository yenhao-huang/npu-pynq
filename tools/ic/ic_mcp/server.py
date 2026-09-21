"""MCP server -- a thin client, nothing more.

Every tool exposed here is generated from the registry, including its JSON
Schema, which is the categories' pydantic models verbatim. There is no tool
list in this file and no per-tool code, so the MCP surface cannot fall behind
the tools it exposes.

Results are JSON, and they are the same JSON the CLI prints and the daemon
returns, because all three call `ic_core.dispatch`. With `IC_DAEMON_URL` set,
calls are forwarded to a running daemon so that a long synthesis survives the
client disconnecting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ic_core import paths
from ic_core.errors import IcError
from ic_core.registry import iter_ops

#: Stay inside the tool timeout of the clients this was tested against
#: (Claude Code and Codex both allow well over a minute). A job that outlives
#: it comes back as a job_id the agent can wait on again, so a slow synthesis
#: degrades into more calls rather than a failure.
DEFAULT_WAIT_S = float(os.environ.get("IC_MCP_WAIT_S", "90"))


def _run(op_name: str, arguments: dict, daemon: str | None, cwd: Path) -> Any:
    if daemon:
        from ic_cli.main import _over_http

        return _over_http(op_name, arguments, daemon, DEFAULT_WAIT_S)
    from ic_core.dispatch import dispatch

    return dispatch(op_name, arguments, cwd=cwd)


def _tool_description(op) -> str:
    description = op.summary
    if op.long_running:
        description += (
            " This can take minutes; if it does, the result is a job_id to wait "
            "on rather than the finished output."
        )
    return description


def _as_function(op, daemon: str | None, root: Path):
    """Build a callable whose signature *is* the category's input model.

    The MCP SDK derives a tool's JSON Schema from the handler's signature, so
    rather than hand-writing a schema per tool -- which would be the one place
    this toolchain could drift from its own contracts -- the signature is
    synthesised from `op.In.model_fields`. The schema an agent sees is
    therefore the category model, by construction.
    """
    import inspect
    from typing import Annotated

    from pydantic import Field

    parameters = []
    annotations: dict[str, Any] = {}
    for name, field in op.In.model_fields.items():
        # A fresh FieldInfo carrying only the description. Reusing the model's
        # own FieldInfo would hand pydantic both a default_factory (from the
        # model) and a default (from the signature), which it rejects.
        annotations[name] = Annotated[field.annotation, Field(description=field.description)]
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=(
                    inspect.Parameter.empty
                    if field.is_required()
                    else field.get_default(call_default_factory=True)
                ),
                annotation=annotations[name],
            )
        )

    def handler(**kwargs):
        given = {k: v for k, v in kwargs.items() if v is not None}
        try:
            return _run(op.name, given, daemon, root)
        except IcError as exc:
            return exc.to_dict()
        except Exception as exc:  # never take the client down with us
            return {"error": {"code": "unexpected", "message": str(exc)}}

    handler.__name__ = op.name
    handler.__doc__ = _tool_description(op)
    handler.__signature__ = inspect.Signature(parameters)
    handler.__annotations__ = annotations
    return handler


def build_server():
    """Register every op, against whichever generation of the SDK is installed.

    mcp 2.x exposes `MCPServer` with an `add_tool` that reflects a handler's
    signature; mcp 1.x exposes a low-level `Server` with `list_tools` and
    `call_tool` decorators. Both are supported because the pinned version of
    an agent harness is not ours to choose.
    """
    daemon = os.environ.get("IC_DAEMON_URL")
    root = Path.cwd()

    try:
        from mcp.server import MCPServer
    except ImportError:
        return _build_legacy_server(daemon, root)

    server = MCPServer(
        name="ic-design-tools",
        version="0.1.0",
        instructions=(
            "RTL design tools. Lint after every edit, simulate to get a waveform "
            "handle, then query that handle with first_mismatch/value_at/"
            "value_range/signals instead of reading the waveform itself."
        ),
    )
    for _category, op in iter_ops():
        server.add_tool(
            _as_function(op, daemon, root),
            name=op.name,
            description=_tool_description(op),
            structured_output=False,
        )
    return server


def _build_legacy_server(daemon: str | None, root: Path):
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    server = Server("ic-design-tools")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=op.name,
                description=_tool_description(op),
                inputSchema=op.In.model_json_schema(),
            )
            for _category, op in iter_ops()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            result = _run(name, arguments or {}, daemon, root)
        except IcError as exc:
            result = exc.to_dict()
        except Exception as exc:
            result = {"error": {"code": "unexpected", "message": str(exc)}}
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    return server


def main() -> int:
    server = build_server()
    if hasattr(server, "run_stdio_async"):
        import anyio

        anyio.run(server.run_stdio_async)
        return 0

    import asyncio

    async def serve() -> None:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
