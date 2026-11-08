#!/usr/bin/env bash
# baseline + single-round: 3 repetitions per problem. Runs the full set, then re-runs any
# (problem, rep) slot whose attempt hit an API/usage-limit error, waiting out the limit.
# Env: AB_RESULTS (data root, default results), AB_EXTRA (extra src.main flags, e.g. --shell).
set -u
cd "$(dirname "$0")/.."
model=$1
R=${AB_RESULTS:-results}; mkdir -p "$R/logs"
for pass in $(seq 1 30); do
  todo=$(AB_RESULTS=$R python3 analysis/affected_single.py "$model")
  [ -z "$todo" ] && { echo "$model single clean after pass $pass" >> "$R/logs/status.txt"; exit 0; }
  while read -r arm reps probs; do
    echo "$model single pass $pass: $arm reps=$reps $(echo $probs | wc -w) problems" >> "$R/logs/status.txt"
    [ "$pass" = 1 ] && [ "$(echo $probs | wc -w)" = 50 ] && probs=""   # first pass: whole set, both arms shuffled together below
    if [ -z "$probs" ]; then
      python3 -m src.main --model "$model" --effort medium --repetitions 3 --rounds 1 --budget 3.00 \
        --parallel 6 --arms notools tools ${AB_EXTRA:-} --output "$R/$model/single" >> "$R/logs/$model.single.log" 2>&1
      break
    fi
    python3 -m src.main $probs --model "$model" --effort medium --repetitions "$reps" --rounds 1 --budget 3.00 \
      --parallel 6 --arms "$arm" ${AB_EXTRA:-} --output "$R/$model/single" >> "$R/logs/$model.single.log" 2>&1
  done <<< "$todo"
  if grep -qhiE "session limit|usage limit|rate limit|quota" $(ls -td $R/$model/single/*/ | head -1)*/round*.jsonl 2>/dev/null; then
    echo "$model single pass $pass hit usage limit; sleeping 30m" >> "$R/logs/status.txt"; sleep 1800
  fi
done
echo "$model single NOT clean after 30 passes" >> "$R/logs/status.txt"
