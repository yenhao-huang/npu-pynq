"""Errors that cross the tool boundary.

Every failure an agent can act on is an `IcError` with a stable `code`. The
daemon maps it to an HTTP 4xx/5xx body, the CLI to a non-zero exit, and the MCP
server to an error result -- all carrying the same code and message.
"""

from __future__ import annotations


class IcError(Exception):
    code = "ic_error"
    http_status = 500

    def __init__(self, message: str, **details: object) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class UnknownOp(IcError):
    code = "unknown_op"
    http_status = 404


class UnknownBackend(IcError):
    code = "unknown_backend"
    http_status = 404


class BackendUnavailable(IcError):
    """The backend executable is not installed or not on PATH."""

    code = "backend_unavailable"
    http_status = 503


class InvalidInput(IcError):
    code = "invalid_input"
    http_status = 422


class HandleNotFound(IcError):
    """A `run_id/filename` handle does not resolve to a file on disk."""

    code = "handle_not_found"
    http_status = 404


class ToolFailed(IcError):
    """The backend ran but could not produce a result (crash, timeout)."""

    code = "tool_failed"
    http_status = 500
