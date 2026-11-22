# The k experiment: what changes and what stays the same

k is the number of animals in B that look exactly the same, the animal itself included.
k = 1 means every tested animal can be picked out in B. k = 8 means all 8 animals of
a tested group look the same in B, so at best you can guess one of 8.

If transfer only works through structure, the best possible score is 1/k and chance is 1/8.

## What changes with k

Only the 16 tested groups change, and only in two places.

**1. Their attribute values (color, size, tail, legs).**
In a tested group the 8 animals form 8/k look-alike sets of k animals each. Animals in
the same set have the same 4 values, and different sets have different values.

| k | sets per tested group | animals per set |
|---|---|---|
| 1 | 8 | 1 |
| 2 | 4 | 2 |
| 4 | 2 | 4 |
| 8 | 1 | 8 |

For k = 1 the tested groups use the same code as the untested groups. For k > 1 the
codes are searched (`tested_codes` in `world/build.py`) so that, over all tested
groups, every value word is used exactly as often as with k = 1. So color_0 appears
160 times in every world, legs_2 32 times, and so on.

**2. Their nests (only with `--nests untested`, which is what the k worlds use).**
Each look-alike set gets one pair of partner animals, and all its members nest near
that same pair. With k = 1 each tested animal has its own pair, and with k = 8 the whole
group shares one pair. If members of a set had different partners, the nests would
tell them apart and the real k would be smaller than we claim.

Partners always come from untested groups, and nobody nests near a tested animal.
So nests cannot break a look-alike set from the other side either.

## What stays exactly the same

Checked by comparing the four built worlds fact by fact:

- the food web, the groups, and which groups are tested
- all food (eats) facts
- all group tag facts
- all hearts facts, including the 128 held-out ones (same animal, same value)
- all facts of untested animals: their attributes, their hearts, the nests they make
- the vocabularies of A and B (every word has the same token in every world)
- the number of facts of every relation, and so the number of sentences
  (A 117,248, B 113,152)
- how often every attribute value word appears
- training settings and seeds

## One side effect to know about

The nests that tested animals make point to untested animals. When k grows, fewer
different untested animals get picked, and those that are picked get picked more often:

| k | tested-to-untested nest facts | different partner animals | most picks of one animal |
|---|---|---|---|
| 1 | 256 | 113 | 8 |
| 2 | 256 | 83 | 6 |
| 4 | 256 | 57 | 8 |
| 8 | 256 | 31 | 16 |

So "who nests near an untested animal" is not identical across k. The untested animals'
own facts are the same, and every untested animal stays unique in B (checked, see below).
This cannot be avoided as long as a whole look-alike set shares its partners.

## How we check that k is what we say

`world/build.py` runs colour refinement (`world/refine.py`) on the B facts of every world
before it saves anything. The build stops if any of these fails:

- every tested look-alike set is exactly one colour class: its members cannot be told
  apart, and they can be told apart from everything else
- every untested animal has its own colour
- every value word, hearts value, tag and relation word has its own colour

The first point is the one that matters for k. It checks both ways: k animals that look
the same, and no extra animal that also looks the same.

## How to build the four worlds

```
python world/web.py --out worlds/web32.json
for k in 1 2 4 8; do
  python world/build.py --web worlds/web32.json --k $k --tags --indiv_repeat 4 \
    --untested_hearts 4,6,9,12,16,20,26,35 --nests untested --out_dir worlds/nestk_k$k
done
```

Or `scripts/make_worlds.sh`, which builds these and the ablation worlds.

## Results (32 groups, mean ± SD over 3 seeds)

| k | best possible 1/k | exact alignment | LM head | probe A to B | probe B |
|---|---|---|---|---|---|
| 1 | 1.00 | 0.95 ± 0.04 | 0.305 ± 0.034 | 0.380 ± 0.100 | 0.302 ± 0.032 |
| 2 | 0.50 | 0.48 ± 0.03 | 0.206 ± 0.023 | 0.232 ± 0.052 | 0.206 ± 0.030 |
| 4 | 0.25 | 0.27 ± 0.03 | 0.154 ± 0.016 | 0.193 ± 0.016 | 0.151 ± 0.016 |
| 8 | 0.125 | 0.09 ± 0.03 | 0.112 ± 0.009 | 0.112 ± 0.012 | 0.115 ± 0.012 |

Chance is 0.125. With only 128 held-out facts per model, one model's accuracy near
chance has a sampling error of about ± 0.03, so single seeds at k = 4 and 8 say little. Pooled z of the LM head over 3 seeds (128 held-out facts each):
k = 1: 10.6, k = 2: 4.8, k = 4: 1.7, k = 8: -0.8.
Exact alignment follows 1/k closely, and transfer goes down with k at every readout.
Per-model numbers are in `results/all_models.csv`.
