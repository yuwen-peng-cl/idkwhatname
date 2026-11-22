#!/bin/zsh
# Ablation at k = 1: what tells animals apart inside a group. 3 worlds x 3 seeds.
# Usage: nohup caffeinate -dims zsh scripts/run_ablation.sh &
cd "${0:A:h}/.."
for w in attronly32_k1 nestonly32_k1 nest32_k1; do
  for s in 42 43 44; do
    zsh scripts/run_model.sh $w $s
  done
done
${PY:-python} scripts/collect.py
