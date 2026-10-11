#!/usr/bin/env bash
# Multi-round uses one clean attempt per problem (no repetitions). Re-run problems that still
# lack one (usage-limit casualties), waiting out the limit, until every problem has one.
# Env: AB_RESULTS (data root, default results), AB_EXTRA (extra src.main flags, e.g. --shell).
set -u
cd "$(dirname "$0")/.."
model=$1
R=${AB_RESULTS:-results}; mkdir -p "$R/logs"
for pass in $(seq 1 30); do
  todo=$(AB_RESULTS=$R python3 analysis/affected.py "$model")
  [ -z "$todo" ] && { echo "$model multi clean after pass $pass" >> "$R/logs/status.txt"; exit 0; }
  echo "$model multi pass $pass: $(echo $todo | wc -w) problems" >> "$R/logs/status.txt"
  python3 -m src.main $todo --model "$model" --effort medium --repetitions 1 --rounds 5 \
    --budget 3.00 --parallel 6 --arms tools ${AB_EXTRA:-} --output "$R/$model/multi" \
    >> "$R/logs/$model.multi.rerun.log" 2>&1
  if grep -qhiE "session limit|usage limit|rate limit|quota" $(ls -td $R/$model/multi/*/ | head -1)*/round*.jsonl 2>/dev/null; then
    echo "$model pass $pass hit usage limit; sleeping 30m" >> "$R/logs/status.txt"; sleep 1800
  fi
done
echo "$model multi NOT clean after 30 passes" >> "$R/logs/status.txt"
