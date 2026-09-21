"""Run persistence: atomicity, handles, the three retention levels."""

from __future__ import annotations

import json

import pytest

from ic_core.errors import HandleNotFound


def test_run_is_invisible_until_it_finishes(store):
    with store.begin(category="sim", op="sim", backend="verilator", inputs={"top": "x"}) as run:
        (run.artifacts / "wave.fst").write_bytes(b"x" * 16)
        # Still a `.tmp` directory: nothing can mistake it for a result yet.
        assert run.dir.name.endswith(".tmp")
        with pytest.raises(HandleNotFound):
            store.resolve(run.handle("wave.fst"))
        run.finish(state="succeeded", out={"ok": True}, duration_s=1.0)
    assert not run.dir.name.endswith(".tmp")
    assert store.resolve(f"{run.run_id}/wave.fst").read_bytes() == b"x" * 16


def test_interrupted_run_leaves_a_tmp_directory_not_a_result(store):
    with pytest.raises(RuntimeError):
        with store.begin(category="sim", op="sim", backend="verilator", inputs={}) as run:
            run_id = run.run_id
            raise RuntimeError("backend died")
    assert not list(store.root.glob(f"*/*-{run_id}-sim"))
    assert list(store.root.glob(f"*/*-{run_id}-sim.tmp"))


def test_gc_clears_tmp_and_work_but_keeps_the_record(store):
    with store.begin(category="sim", op="sim", backend="verilator", inputs={}) as run:
        (run.work / "obj_dir").mkdir()
        (run.artifacts / "wave.fst").write_bytes(b"data")
        run.finish(state="succeeded", out={"ok": True}, duration_s=1.0)
    removed = store.gc()
    assert removed["work"] == 1
    assert not (store.run_dir(run.run_id) / "work").exists()
    # meta.json survives reclaiming, which is the whole point of the split.
    assert store.meta(run.run_id)["state"] == "succeeded"
    assert store.resolve(f"{run.run_id}/wave.fst").exists()


def test_input_files_are_hashed_not_copied(store, tmp_path):
    source = tmp_path / "npu_pe.sv"
    source.write_text("module npu_pe; endmodule\n")
    with store.begin(category="lint", op="lint", backend="verilator",
                     inputs={"top": "npu_pe"}, input_files=[str(source)]) as run:
        run.finish(state="succeeded", out={"ok": True}, duration_s=0.1)
    recorded = store.meta(run.run_id)["inputs"]["files"][0]
    assert recorded["path"] == str(source)
    assert len(recorded["sha256"]) == 64
    assert not (store.run_dir(run.run_id) / "npu_pe.sv").exists()


def test_index_is_rebuildable_from_the_tree(store):
    with store.begin(category="lint", op="lint", backend="verilator",
                     inputs={"top": "npu_pe"}) as run:
        run.finish(state="succeeded", out={"ok": True}, duration_s=0.5, top="npu_pe")
    store.db_path.unlink()
    assert store.rebuild_index() == 1
    assert store.list_runs(top="npu_pe")[0]["run_id"] == run.run_id


def test_handle_survives_an_index_that_is_gone(store):
    with store.begin(category="sim", op="sim", backend="verilator", inputs={}) as run:
        (run.artifacts / "sim.log").write_text("PASS\n")
        run.finish(state="succeeded", out={"ok": True}, duration_s=0.1)
    store.db_path.unlink()
    assert store.resolve(f"{run.run_id}/sim.log").read_text() == "PASS\n"


def test_meta_records_what_ran(store):
    with store.begin(category="synth", op="synth", backend="yosys",
                     backend_version="Yosys 0.23", inputs={"top": "npu_pe"}) as run:
        run.finish(state="succeeded", out={"ok": True}, exit_code=0, duration_s=2.0)
    meta = json.loads((store.run_dir(run.run_id) / "meta.json").read_text())
    assert meta["backend_version"] == "Yosys 0.23"
    assert meta["exit_code"] == 0 and meta["duration_s"] == 2.0
