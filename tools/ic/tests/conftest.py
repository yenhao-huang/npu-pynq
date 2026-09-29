from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the run store at a temp directory for the duration of a test."""
    monkeypatch.setenv("IC_ROOT", str(tmp_path / "store"))
    from ic_core.runstore import RunStore

    return RunStore()


def needs(executable: str):
    return pytest.mark.skipif(
        shutil.which(executable) is None, reason=f"{executable} not installed"
    )


@pytest.fixture()
def counter_wave(store, tmp_path):
    """Simulate the buggy-counter fixture once and hand back its wave handle."""
    from ic_core import dispatch

    out = dispatch(
        "sim",
        {
            "files": [str(FIXTURES / "counter.sv"), str(FIXTURES / "tb_counter.sv")],
            "tb": "tb_counter",
        },
        cwd=tmp_path,
    )
    assert out["built"], out["summary"]
    assert out["wave"], "sim produced no waveform"
    return out
