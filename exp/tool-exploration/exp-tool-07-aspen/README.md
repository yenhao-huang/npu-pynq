# Tool 07: ppa_pareto

Hypothesis: Retain every measured nondominated trade-off.

Run from the repository root:

```sh
python exp/tool-exploration/run.py --tool 07 --resume
```

Inputs are in config.json and shared fixtures/. Results go to ../output/07.json.
Tools 05-08 use tool 04 measurements; --resume reuses matching input fingerprints.
Refer to docs/goals/0928-tool-exploration/report.md for the recorded result,
limitations, paper attribution, and the next decision.

## Larger CSD integration

```sh
python exp/tool-exploration/wide_workflow.py --tool 07
```

This module uses the existing 16-bit, constant-255 CSD study. Its input and
backend fingerprints scope resumable results under this module's `output/`.
Dependent operations reuse actual measurement records. The complete workflow
checks all 65,536 binary inputs, rejects a highest-output-bit mutation, and
exercises explicit invalid inputs for each operation. See
`docs/goals/0928-tool-exploration/evidence/wide-workflow.json` for this operation's
positive and negative results. Combinational measurements are diagnostic;
the separate registered three-pair study supplies physical acceptance evidence.
