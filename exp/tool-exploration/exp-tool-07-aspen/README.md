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
