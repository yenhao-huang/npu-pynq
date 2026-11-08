"""Hidden evaluation, never fed back into the model conversation."""
from .shipped import grade_shipped
from .heldout import grade_heldout
from .ppa import measure_ppa
from ..problems.patches import HELDOUT_SKIP

class Grader:
    def __init__(self, config):
        self.config = config

    def evaluate(self, problem, solution, work):
        shipped, info_s = grade_shipped(problem, solution, work)
        heldout, info_h = grade_heldout(problem, solution, work)
        skipped = problem.id in HELDOUT_SKIP and heldout is None
        row = dict(heldout_skipped=skipped, passed_shipped=shipped, passed_heldout=heldout,
                   passed=bool(shipped and (heldout is True or skipped)),
                   shipped_info="" if shipped else info_s,
                   heldout_info="" if heldout else info_h)
        if row["passed"]:
            row.update(measure_ppa(problem, solution, work, self.config))
        return row
