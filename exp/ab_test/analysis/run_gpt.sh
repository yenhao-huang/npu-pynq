#!/usr/bin/env bash
# gpt-6-sol (Codex CLI backend) on the same three groups. Run from anywhere.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/logs
python3 -m src.main --model gpt-6-sol --effort medium --repetitions 3 --rounds 1 --budget 3.00 \
  --parallel 6 --arms notools tools --output results/gpt-6-sol/single > results/logs/gpt-6-sol.single.log 2>&1
echo "gpt-6-sol single exit=$?" >> results/logs/status.txt
analysis/rerun_multi.sh gpt-6-sol
echo GPT_DONE >> results/logs/status.txt
