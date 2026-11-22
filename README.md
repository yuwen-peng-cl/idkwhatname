# Relational identifiability in a toy world

Question: two languages share no words. A fact is seen only in language A. Can a model
answer it in language B? And does that depend on how well the animal is defined by its
relations?

We build a small artificial world of animals and write it in two made-up languages
(A: 2 CJK characters per word, B: 3 Hiragana per word, no shared tokens). The hearts
facts of some animals ("tested" animals) are only in A. We ask them in B.

k is the number of animals that look exactly the same in B (the animal itself included).
If the model can only match animals across languages through their relations, the best
it can do on a tested fact is 1/k. Chance is 1/8.

## Folders

```
world/     build the worlds
  refine.py    colour refinement: which animals can be told apart (used to check every web and world)
  web.py       food web where no group can be swapped with another
  build.py     one world: facts, sentences in A and B, vocabularies
train/
  train.py     small BERT (4 layers, hidden 128), masked LM, 100 epochs
eval/
  evaluate.py  all metrics for one model (memory, transfer, alignment)
scripts/
  make_worlds.sh   build the web and all worlds
  run_model.sh     train + evaluate one model
  run_k.sh         k experiment (4 worlds x 3 seeds)
  run_ablation.sh  what tells animals apart inside a group (3 worlds x 3 seeds)
  collect.py       results/models/*.json -> results/all_models.csv + summary tables
notes/
  worlds.md    every world we tried, in order, with results
  k.md         the k experiment: what changes with k and what stays the same
results/
  models/          one json per trained model
  all_models.csv   one row per model
worlds/    built worlds and models (not in git, rebuild with make_worlds.sh)
```

## Run

```
zsh scripts/make_worlds.sh
nohup caffeinate -dims zsh scripts/run_k.sh &
nohup caffeinate -dims zsh scripts/run_ablation.sh &
python scripts/collect.py
```

Set `PY=/path/to/python` if `python` is not the right environment.

## Main result (32 groups, mean ± SD over 3 seeds)

| k | best possible 1/k | exact alignment | LM head | probe A to B | probe B |
|---|---|---|---|---|---|
| 1 | 1.00 | 0.95 ± 0.04 | 0.305 ± 0.034 | 0.380 ± 0.100 | 0.302 ± 0.032 |
| 2 | 0.50 | 0.48 ± 0.03 | 0.206 ± 0.023 | 0.232 ± 0.052 | 0.206 ± 0.030 |
| 4 | 0.25 | 0.27 ± 0.03 | 0.154 ± 0.016 | 0.193 ± 0.016 | 0.151 ± 0.016 |
| 8 | 0.125 | 0.09 ± 0.03 | 0.112 ± 0.009 | 0.112 ± 0.012 | 0.115 ± 0.012 |

Chance is 0.125. See `notes/k.md` for the setup and `notes/worlds.md` for how we got here.

## Metrics

All from `eval/evaluate.py`. Every forced choice and probe has 8 classes (chance 1/8).

- memory: `a_seen` held-out facts asked in A, `b_other` untested animals asked in B
- transfer (held-out facts asked in B):
  `lm_fc` LM head, choose among the 8 hearts values;
  `lm_free` LM head, top-1 over all words;
  `probe_ab` linear probe trained on A, tested on B;
  `probe_b` linear probe trained on untested animals in B
- alignment: for every animal in B, its nearest animal in A (cosine, each language's mean removed).
  `exact` it is the same animal, `group` same food group, `lookalike` same look-alike set
- values: the same for value words (attributes, hearts, tags)
