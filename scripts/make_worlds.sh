#!/bin/zsh
# Build the food web and all worlds used in the k experiment and the ablation.
# Usage: zsh scripts/make_worlds.sh
cd "${0:A:h}/.."
PY=${PY:-python}
H=4,6,9,12,16,20,26,35   # how many untested animals get each hearts value

$PY world/web.py --levels 8,8,8,8 --max_eat 2 --seed 42 --out worlds/web32.json

# k experiment: only the tested groups change with k (see notes/k.md)
for k in 1 2 4 8; do
  $PY world/build.py --web worlds/web32.json --k $k --tags --indiv_repeat 4 --untested_hearts $H \
    --nests untested --out_dir worlds/nestk_k$k
done

# ablation at k = 1: attributes + nests, nests only, attributes only
$PY world/build.py --web worlds/web32.json --k 1 --tags --indiv_repeat 4 --untested_hearts $H \
  --nests any --out_dir worlds/nest32_k1
$PY world/build.py --web worlds/web32.json --k 1 --tags --indiv_repeat 7 --untested_hearts $H \
  --nests any --no_attributes --out_dir worlds/nestonly32_k1
$PY world/build.py --web worlds/web32.json --k 1 --tags --indiv_repeat 7 --untested_hearts $H \
  --out_dir worlds/attronly32_k1
