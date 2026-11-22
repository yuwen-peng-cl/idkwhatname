#!/bin/zsh
# k experiment: 4 worlds x 3 seeds, one model after another, then the summary table.
# Usage: nohup caffeinate -dims zsh scripts/run_k.sh &
cd "${0:A:h}/.."
for k in 1 2 4 8; do
  for s in 42 43 44; do
    zsh scripts/run_model.sh nestk_k$k $s
  done
done
${PY:-python} scripts/collect.py
