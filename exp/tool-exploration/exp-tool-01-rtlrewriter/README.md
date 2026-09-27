# Tool 01: optimization_rules

Hypothesis: Retrieve area rules and their semantic preconditions.

Run from the repository root:

```sh
python exp/tool-exploration/run.py --tool 01 --resume
```

Inputs are in config.json and shared fixtures/. Results go to ../output/01.json.
Tools 05-08 use tool 04 measurements; --resume reuses matching input fingerprints.
Refer to docs/goals/0928-tool-exploration/report.md for the recorded result,
limitations, paper attribution, and the next decision.
