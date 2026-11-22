#!/bin/zsh
# Train one model on one world and evaluate it.
# Usage: zsh scripts/run_model.sh <world name> <seed> [both|b]
# Long runs: nohup caffeinate -dims zsh scripts/run_model.sh nestk_k1 42 &
cd "${0:A:h}/.."
PY=${PY:-python}
W=$1
S=$2
L=${3:-both}
M=worlds/$W/model_${L}_s$S

$PY train/train.py --world_dir worlds/$W --langs $L --seed $S --out_dir $M > $M.train.log 2>&1 || exit 1
$PY eval/evaluate.py --world_dir worlds/$W --model_dir $M/final --out results/models/${W}__model_${L}_s$S.json > /dev/null
