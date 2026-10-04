from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import paths
from llm import LLMBackend
from openroad import ORFSConfig, check_image, run_orfs
from prompt_formatter import DesignSpec, format_prompt, parse_user_prompt
from retrieval import RTLLibrary, retrieve
from synthesis import elaborate, stage_sources
from tpugen_types import PPA, DesignState, FlowError, PPATarget
from validate import check_macros, summarize_elaboration, summarize_orfs_log

from .stop import MaxIterations, StopCondition


@dataclass
class IterationRecord:
    iteration: int
    workdir: Path
    spec: DesignSpec
    vh_path: Path
    file_count: int
    elapsed: float
    ppa: PPA | None = None
    gds: Path | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.ppa is not None and not self.errors


@dataclass
class TPUGenResult:
    run_dir: Path
    iterations: list[IterationRecord]
    stopped_because: str

    @property
    def best(self) -> IterationRecord | None:
        done = [i for i in self.iterations if i.ok]
        return done[-1] if done else None

    def table(self) -> str:
        head = f"{'iter':>4}  {'files':>5}  {'area/um2':>10}  {'WNS/ns':>8}  {'power/W':>9}  {'time/s':>7}  status"
        rows = [head, "-" * len(head)]
        for r in self.iterations:
            if r.ppa:
                rows.append(
                    f"{r.iteration:>4}  {r.file_count:>5}  {r.ppa.area:>10.1f}  "
                    f"{r.ppa.wns:>8.3f}  {r.ppa.total_power:>9.3e}  "
                    f"{r.elapsed:>7.0f}  ok"
                )
            else:
                reason = r.errors[0][:40] if r.errors else "failed"
                rows.append(
                    f"{r.iteration:>4}  {r.file_count:>5}  {'-':>10}  {'-':>8}  "
                    f"{'-':>9}  {r.elapsed:>7.0f}  {reason}"
                )
        return "\n".join(rows)


class TPUGenFlow:
    """Runs the TPU-Gen pipeline, optionally for several rounds.

    Every round writes a self-contained directory under runs/<run-id>/ so
    rounds can be compared afterwards; nothing is overwritten.
    """

    def __init__(
        self,
        llm: LLMBackend,
        *,
        orfs: ORFSConfig | None = None,
        retrieval_mode: str = "rag",
        runs_dir: Path | None = None,
        rtl_dir: Path | None = None,
        top: str = paths.TOP_MODULE,
    ):
        paths.check()
        self.llm = llm
        self.orfs = orfs or ORFSConfig()
        self.retrieval_mode = retrieval_mode
        self.runs_dir = runs_dir or paths.RUNS
        self.library = RTLLibrary.load(rtl_dir or paths.RTL_DIR)
        self.top = top
        check_image(self.orfs.image)

    def run(
        self,
        user_prompt: str,
        target: PPATarget,
        *,
        stop: StopCondition | None = None,
        max_iterations: int = 3,
        run_id: str | None = None,
    ) -> TPUGenResult:
        stop = stop or MaxIterations(max_iterations)
        run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        spec = parse_user_prompt(user_prompt)
        records: list[IterationRecord] = []
        previous_ppa: PPA | None = None
        errors: list[str] = []
        stopped = f"reached {max_iterations} iterations"

        for i in range(max_iterations):
            record, state = self._iterate(
                run_dir, i, user_prompt, spec, target, previous_ppa, errors
            )
            records.append(record)
            (record.workdir / "state.json").write_text(state.to_json(), encoding="utf-8")

            previous_ppa, errors = record.ppa, record.errors
            if stop.met(state):
                stopped = stop.description
                break

        result = TPUGenResult(run_dir=run_dir, iterations=records, stopped_because=stopped)
        (run_dir / "summary.txt").write_text(
            f"{result.table()}\n\nstopped: {stopped}\n", encoding="utf-8"
        )
        return result

    def _iterate(
        self,
        run_dir: Path,
        i: int,
        user_prompt: str,
        spec: DesignSpec,
        target: PPATarget,
        previous_ppa: PPA | None,
        errors: list[str],
    ) -> tuple[IterationRecord, DesignState]:
        started = time.time()
        workdir = run_dir / f"iter_{i:02d}"
        (workdir / "src").mkdir(parents=True, exist_ok=True)

        state = DesignState(
            user_prompt=user_prompt, target=target, iteration=i, workdir=workdir
        )

        state.formatted_prompt = format_prompt(
            spec, target, previous_ppa=previous_ppa, errors=errors
        )
        (workdir / "prompt.txt").write_text(state.formatted_prompt, encoding="utf-8")

        state.vh_text = self.llm.generate(state.formatted_prompt)
        vh_path = workdir / "generated.vh"
        vh_path.write_text(state.vh_text, encoding="utf-8")

        macro_problems = check_macros(state.vh_text)
        if macro_problems:
            record = IterationRecord(
                iteration=i, workdir=workdir, spec=spec, vh_path=vh_path,
                file_count=0, elapsed=time.time() - started,
                errors=macro_problems,
            )
            state.errors = macro_problems
            return record, state

        retrieved = retrieve(
            state.vh_text, self.library, self.top, mode=self.retrieval_mode
        )
        state.filelist = retrieved.filelist
        state.config = retrieved.defines
        (workdir / "retrieval.json").write_text(
            json.dumps(
                {
                    "mode": retrieved.mode,
                    "modules": retrieved.modules,
                    "files": [p.name for p in retrieved.filelist],
                    "missing": retrieved.missing,
                    "library_size": len(self.library.module_file),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        record = IterationRecord(
            iteration=i,
            workdir=workdir,
            spec=spec,
            vh_path=vh_path,
            file_count=len(retrieved.filelist),
            elapsed=0.0,
        )

        if retrieved.missing:
            record.errors = [f"module not in RTL library: {m}" for m in retrieved.missing]
            state.errors = record.errors
            record.elapsed = time.time() - started
            return record, state

        stage_sources(retrieved.filelist, state.vh_text, workdir / "src")

        elab = elaborate(workdir / "src", self.top)
        if not elab.ok:
            record.errors = summarize_elaboration(elab.log) or ["elaboration failed"]
            state.errors = record.errors
            record.elapsed = time.time() - started
            return record, state

        try:
            orfs_result = run_orfs(workdir, self.orfs)
        except FlowError as exc:
            log = workdir / "orfs.log"
            record.errors = summarize_orfs_log(log) if log.exists() else [str(exc)]
            state.errors = record.errors
            record.elapsed = time.time() - started
            return record, state

        record.ppa = state.ppa = orfs_result.ppa
        record.gds = state.gds = orfs_result.gds
        record.elapsed = time.time() - started
        return record, state
