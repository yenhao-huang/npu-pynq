#!/usr/bin/env bash
# Full A/B run per docs/goals/ab_test_human.md. Run from exp/ab_test/.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/logs
run() {  # model group rounds arms...
  local model=$1 group=$2 rounds=$3; shift 3
  python3 -m src.main --model "$model" --effort medium --repetitions 3 --rounds "$rounds" \
    --budget 3.00 --parallel 6 --arms "$@" --output "results/$model/$group" \
    > "results/logs/$model.$group.log" 2>&1
  echo "$model $group exit=$?" >> results/logs/status.txt
}
run claude-opus-5-5   single 1 notools tools &
run claude-sonnet-5-5 single 1 notools tools &
wait
# multi-round: one attempt per problem (no repetitions); rerun_multi.sh re-runs problems whose
# attempt hit the account usage limit, until each problem has one clean attempt.
analysis/rerun_multi.sh claude-opus-5-5 &
analysis/rerun_multi.sh claude-sonnet-5-5 &
wait
echo ALL_DONE >> results/logs/status.txt
