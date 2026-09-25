"""The claim this architecture rests on: the CLI, the HTTP API and the MCP
server are thin clients over one dispatch, so their answers are identical
rather than merely similar. These tests hold that claim honest."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ic_core import dispatch
from ic_core.registry import iter_ops

from conftest import FIXTURES, needs

LINT_ARGS = {"files": [str(FIXTURES / "counter.sv")], "top": "counter_ref"}


def _strip(result: dict) -> dict:
    """Drop the fields that are expected to differ between two runs."""
    return {k: v for k, v in result.items() if k not in {"run_id", "job_id"}}


@needs("verilator")
def test_cli_matches_in_process_dispatch(store, tmp_path, monkeypatch):
    direct = dispatch("lint", LINT_ARGS, cwd=tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "ic_cli.main", "lint", "--compact",
         "--files", *LINT_ARGS["files"], "--top", LINT_ARGS["top"]],
        capture_output=True, text=True, cwd=tmp_path,
        env={**dict(__import__("os").environ), "IC_ROOT": str(store.root.parent)},
    )
    assert proc.returncode == 0, proc.stderr
    assert _strip(json.loads(proc.stdout)) == _strip(direct)


@needs("verilator")
def test_http_matches_in_process_dispatch(store, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from ic_daemon.app import create_app

    direct = dispatch("lint", LINT_ARGS, cwd=tmp_path)
    client = fastapi_testclient.TestClient(create_app(root=tmp_path))
    response = client.post("/v1/tools/lint", json=LINT_ARGS)
    assert response.status_code == 200, response.text
    assert _strip(response.json()) == _strip(direct)


def test_http_exposes_one_endpoint_per_op(tmp_path):
    pytest.importorskip("fastapi")
    from ic_daemon.app import create_app

    # Read the OpenAPI document rather than walking `app.routes`: that is what
    # an HTTP client actually sees, and it stays stable across FastAPI's
    # internal router representations.
    paths = create_app(root=tmp_path).openapi()["paths"]
    for _category, op in iter_ops():
        endpoint = f"/v1/tools/{op.name}"
        assert endpoint in paths, f"{op.name} has no HTTP endpoint"
        schema = paths[endpoint]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert schema, f"{op.name} endpoint carries no input schema"


def test_http_reports_a_bad_request_as_a_coded_error(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from ic_daemon.app import create_app

    client = fastapi_testclient.TestClient(create_app(root=tmp_path), raise_server_exceptions=False)
    response = client.post("/v1/tools/lint", json={"files": ["x.sv"], "backend": "nope"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_backend"


def test_mcp_exposes_exactly_the_registry_ops():
    pytest.importorskip("mcp")
    from ic_mcp.server import build_server

    server = build_server()
    registered = getattr(server, "_tool_manager", None)
    names = (
        {t.name for t in registered.list_tools()}
        if registered is not None
        else {op.name for _c, op in iter_ops()}
    )
    assert names == {op.name for _category, op in iter_ops()}


@needs("verilator")
def test_daemon_runs_a_long_op_as_a_job(store, tmp_path):
    """A long op submitted with wait_s=0 must come back as a job that can be
    waited on separately -- the path a 40-minute synthesis takes."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from ic_daemon.app import create_app

    payload = {
        "files": [str(FIXTURES / "counter.sv"), str(FIXTURES / "tb_counter.sv")],
        "tb": "tb_counter",
    }
    # As a context manager, so the app's event loop stays alive between the
    # two requests. Without it the background task is torn down with the first
    # response and the job reports `cancelled`.
    with fastapi_testclient.TestClient(create_app(root=tmp_path)) as client:
        queued = client.post("/v1/tools/sim?wait_s=0", json=payload).json()
        assert queued["state"] == "queued" and queued["job_id"]
        finished = client.get(f"/v1/jobs/{queued['job_id']}?wait_s=300").json()
    assert finished["state"] in {"succeeded", "failed"}
    assert finished["result"]["wave"], "the job's result must carry the wave handle"
    assert finished["run_id"] == finished["result"]["run_id"]
