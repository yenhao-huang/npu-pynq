# Tool 09: ppa_select

Hypothesis: Choose an affordable rewrite while retaining exploration of unvisited actions.

Run from the repository root:

```sh
python exp/tool-exploration/run.py --tool 09 --resume
```

Inputs are in config.json and shared fixtures/. Results go to ../output/09.json.
Tools 05-08 use tool 04 measurements; --resume reuses matching input fingerprints.
Refer to docs/goals/0928-tool-exploration/report.md for the recorded result,
limitations, paper attribution, and the next decision.

## Larger CSD integration

```sh
python exp/tool-exploration/wide_workflow.py --tool 09
```

This module uses the existing 16-bit, constant-255 CSD study. Its input and
backend fingerprints scope resumable results under this module's `output/`.
Dependent operations reuse actual measurement records. The complete workflow
checks all 65,536 binary inputs, rejects a highest-output-bit mutation, and
exercises explicit invalid inputs for each operation. See
`docs/goals/0928-tool-exploration/evidence/wide-workflow.json` for this operation's
positive and negative results. Combinational measurements are diagnostic;
the separate registered three-pair study supplies physical acceptance evidence.

## Finite score controls

Run `python exp/tool-exploration/exp-tool-09-rtlrewriter/controls.py` to reuse the
larger CSD study's reward/cost and reject three derived-score overflow cases.
Finite inputs do not guarantee finite arithmetic results. Invalid scores must
raise an explicit input error before selection, including huge visit counts.
This runner performs no new physical measurement. See the goal's
`evidence/selector-finite-controls.json` and `semantic-review.md`.
