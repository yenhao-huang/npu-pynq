# IC tools A/B experiment

This study compares no IC tools, one round with IC tools, and five rounds with
IC tools on 50 RTLLM v2.0 tasks using Claude Opus 5.5, Claude Sonnet 5.5, and
gpt-6-sol. [Report.md](Report.md) contains the results, including the original
power, area, and delay measurements. [Reproduce.md](Reproduce.md) describes the
environment, commands, and limits of reproduction.

The repository contains the experiment harness (`src/`), setup and analysis
scripts, tests, the MIT-licensed task inputs (`problems/`), and selected
acceptance examples (`docs/`). The full model output, attempt records, and RTL
snapshots are retained only in the local `results/` archive. They are not part
of this pull request. This keeps the published experiment below 1 MB of file
content while retaining the methods and reported findings.

From this directory, check the source and task catalog with:

```bash
python3 -m unittest discover -s tests
python3 -m src.main --list
```

The analysis scripts read a fresh or locally retained `results/` directory.
They cannot independently recalculate the historical report from this compact
checkout alone. See [Reproduce.md](Reproduce.md) to run the experiment again.
