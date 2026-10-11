#!/usr/bin/env bash
# Shell variant: every arm gets a shell (Bash); arms differ only by the ic MCP tools.
# Same groups as run_all.sh / run_gpt.sh; data under results/shell/. Run from anywhere.
set -u
cd "$(dirname "$0")/.."
export AB_RESULTS=results/shell AB_EXTRA=--shell
mkdir -p "$AB_RESULTS/logs"
for model in claude-opus-5-5 claude-sonnet-5-5 gpt-6-sol; do
  (analysis/rerun_single.sh "$model" && analysis/rerun_multi.sh "$model") &
done
wait
echo SHELL_ALL_DONE >> "$AB_RESULTS/logs/status.txt"
