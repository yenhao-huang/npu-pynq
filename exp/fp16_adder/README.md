# FP16 adder evaluation

The active experiment implements binary16 addition, matching the operation and
width of RTLScout Section 7.1. Historical FP12 multiplier results in the sibling
folder do not count toward this goal.

- `generate.py`, `architectures.py`: exact baseline and compact candidate RTL.
- `model/reference.py`, `test/`: independent arithmetic reference and checks.
- `run_eval.py`, `sta.py`: RTL verification, mapping, gate verification, Docker STA.
- `campaign.py`: hypotheses and architecture/constraint exploration.
- `summarize.py`: evidence-derived report and distinct-design joint ranking.
- `reproduce.py`: frozen top-three RTL replay with exact measurement comparison.
- `docs/contract.md`: numeric semantics, PPA assumptions and coverage.

See `report.md` for current measured results and `reproduce.md` for commands.
