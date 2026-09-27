# Tool 02: width_advice

Hypothesis: A 16-bit result holding the sum of two 4-bit unsigned values needs only 5 bits.

Run from the repository root:

```sh
python exp/tool-exploration/run.py --tool 02 --resume
```

Inputs are in config.json and shared fixtures/. Results go to ../output/02.json.
Tools 05-08 use tool 04 measurements; --resume reuses matching input fingerprints.
Refer to docs/goals/0928-tool-exploration/report.md for the recorded result,
limitations, paper attribution, and the next decision.
