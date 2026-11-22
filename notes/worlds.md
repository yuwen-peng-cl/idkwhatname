# All worlds we tried

This file lists every world we built, in the order we built them. For each one: what the world looks like, what we changed, the settings, and what came out.

## The question

Two languages (A and B) describe the same world but share no tokens. Some facts are only in A. Can the model answer them in B?

The test fact is always the number of hearts of an animal. For tested animals the hearts fact is only in A. We ask it in B.

k = how many animals look exactly the same in B (counting the animal itself). k = 1 means every animal can be told apart in B. k = 8 means eight animals look the same, so the model can at best guess one of eight.

## Things that are the same everywhere

- Model: small BERT, 128 dim, 4 layers, 4 heads, trained from scratch with masked language modelling.
- Languages: A uses 2-character CJK tokens, B uses 3-character Hiragana tokens. No token is shared.
- Every fact becomes 8 English template sentences (for example "X has Y hearts", "X owns Y hearts"). Then every word is replaced by its A or B token.
- Training: lr 3e-4, 100 epochs, batch 64, no weight decay, dropout 0.1, linear lr decay. Exceptions are noted.
- One seed (42) unless the row says otherwise.

## Metrics

All numbers below come from one script, `eval/evaluate.py`, run on every model. Chance is 0.125 for everything that picks one of the 8 hearts values.

- **memory A / B**: hearts facts the model has seen, asked in A (tested animals) and in B (untested animals). Both should be near 1.
- **exact**: for each tested animal in B, is the nearest animal in A (cosine, static embeddings) the same animal?
- **group**: same as exact, but it is enough to land in the right food group.
- **hearts / values**: for each hearts value (and each attribute value), is the nearest A word the right one? Written as "aligned / total".
- **LM head**: ask "X has [MASK] hearts" in B for a tested animal and pick the most likely of the 8 hearts values. This is transfer at the output layer.
- **probe AB**: a linear classifier trained on the [MASK] state of the tested facts in A, applied to the same facts in B (layer 4). This is the method of the old project.
- **probe B**: a linear classifier trained on the [MASK] state of untested animals in B, applied to the tested animals in B (layer 4). This one can also be run on a B-only model.
- **B-only**: a model trained on B only. Its LM head and probe B show how much can be guessed without A.

---

## 1. Attribute table

**World:** 1000 entities, no links between entities. Each entity has 4 attributes (arms, habitat, color, size, with 5/6/7/9 values) and a hearts value.

**Example:** `(e000, arms, arms_2)`, `(e000, habitat, habitat_4)`, `(e000, color, color_5)`, `(e000, size, size_8)`, `(e000, hearts, hearts_6)` (only in A).

**What we changed:** first world.

**Settings:** lr 1e-3 first, then lr 3e-4.

**Result:**
- lr 1e-3: nothing learned (memory 0.12 / 0.16).
- lr 3e-4: everything memorized, but the two languages do not line up at all. exact 0.00, every attribute value and hearts value 0 aligned. LM head 0.15, probe AB 0.11.

**Takeaway:** without links between entities, the languages do not line up.

---

## 2. Old food chain

**World:** 1024 animals in 128 groups of 8, 4 levels. Every group eats exactly one group of the level below (all members eat all members). 4 attributes (color, legs, size, skin, with 6/7/9/10 values). Every group gets its own attribute values. Hearts: in every group the 8 animals have the 8 different values.

**Example:** `(a0000, color, color_2)`, `(a0000, legs, legs_3)`, `(a0000, size, size_6)`, `(a0000, skin, skin_7)`, `(a0000, hearts, hearts_6)` (only in A).

**What we changed:** animals now eat each other.

**Settings:** k = 1, 2, 4, 8, bilingual and B-only for each.

**Result:**

| k | exact | group | hearts | LM head | probe AB |
|---|---|---|---|---|---|
| 1 | 0.11 | 0.26 | 0/8 | 0.12 | 0.15 |
| 2 | 0.04 | 0.17 | 3/8 | 0.13 | 0.13 |
| 4 | 0.03 | 0.22 | 2/8 | 0.13 | 0.13 |
| 8 | 0.03 | 0.24 | 4/8 | 0.12 | 0.11 |

B-only LM head: 0.12 to 0.13.

**Takeaway:** some lining up appears once animals are linked, but it is weak and there is no transfer.

### 2b. Plus order facts between values

**What we changed:** every pair of values gets an order fact, for example `(legs_1, below, legs_4)` (145 facts instead of 35).

**Result:** exact 0.02, group 0.11, hearts 1/8, LM head 0.12. No better.

### 2c. 8 attributes

**What we changed:** 8 attributes (5 to 13 values). Animals in a group differ in at least 4 attributes.

**Result:** exact 0.08, group 0.21, hearts 3/8, LM head 0.13. No better.

### 2d. 20 attributes

**What we changed:** 20 attributes with 3 to 6 values each.

**Result:** stopped at epoch 80. Hearts never memorized, group lining up about 0.06.

---

## 3. Food web where every group is unique

### Pilot: web only

**World:** 32 groups (4 levels x 8) of 8 identical animals. Every group eats 1 to 2 random groups of the level below. We redraw the web until no group can be swapped with another (checked with colour refinement and with networkx). No attributes, no hearts.

**Example:** `(a0080, eats, a0048)`.

**Result:** group 1.00, exact 0.14 (about 1/8, which is expected because the 8 animals of a group are identical). With 16 groups: group 0.98, exact 0.17.

**Takeaway:** a web where every group is unique lines up perfectly at the group level.

### 3a. Big web + shared attributes

**World:** 128 groups (32/48/32/16). All groups use the same 4 attributes (color, size, tail, legs). Inside untested groups the counts are fixed (color 5+3, size 6+2, tail 7+1, legs 4+3+1), so values can be told apart by how often they occur.

**Example:** `(a0024, color, color_0)`, `(a0024, size, size_0)`, `(a0024, tail, tail_1)`, `(a0024, legs, legs_0)`, `(a0024, hearts, hearts_3)` (only in A).

**Result:**
- k = 1: nothing about single animals learned (memory 0.13 / 0.12). exact 0.01, group 0.06.
- k = 8: hearts memorized only after epoch 70 (memory 0.72 / 0.98). exact 0.02, group 0.26.
- The difference between k = 1 and k = 8 came from training luck, not from k.

### 3b. 32-group pilot web + shared attributes

**What we changed:** the 32-group pilot web, plus the shared attributes and hearts.

**Result:** group 0.98, but hearts never memorized (memory 0.12 / 0.12). Facts about single animals were only about 20% of the sentences an animal appears in.

### 3c. Fewer food sentences

**What we changed:** food facts use 2 templates instead of 8.

**Result:** still not memorized (memory 0.20 / 0.17), group dropped to 0.73. This did not add sentences about single animals, it only removed food sentences.

### 3d. Attribute sentences repeated 4 times (32 groups)

**What we changed:** attribute and hearts sentences are repeated 4 times, so each animal appears about as often in them as in food sentences.

**Result:** memorized (1.00 / 1.00). group 0.86, but exact 0.05, attribute values mostly not lined up, hearts 0/8. LM head 0.08.

### 3e. Same on 16 groups

**Result:** memorized. exact 0.28, group 0.62, all attribute values lined up, hearts 1/8. LM head 0.17 (B-only 0.14).

**Takeaway:** shared attributes help animals and values line up, but they pull groups together.

---

## 4. Every group owns its own set of attributes (16 groups)

**World:** 7 attributes. Every group has its own 4 of them, so the set of attributes already tells the group.

**Example:** `(a0008, claws, claws_0)`, `(a0008, teeth, teeth_2)`, `(a0008, ears, ears_0)`, `(a0008, mane, mane_1)`, `(a0008, hearts, hearts_5)` (only in A).

**Result:**
- 4 of 7: group 1.00, exact 0.19, most values not lined up, hearts 0/8. LM head 0.12 (B-only 0.11).
- 5 of 7: group 1.00, exact 0.17, hearts 0/8. LM head 0.12 (B-only 0.16).

**Takeaway:** groups line up, values and single animals do not.

---

## 5. Group tags

### 5a. 8 groups + a tag fact

**World:** 8 groups (4 levels x 2), shared attributes. Every animal gets a fact "animal is tag_g", repeated once for each food fact of the animal.

**Example:** `(a0008, color, color_0)`, ..., `(a0008, kind, tag_1)`, `(a0008, hearts, hearts_3)` (only in A).

**Result:** exact 0.44, group 0.88, tags 7/8, hearts 2/8. LM head 0.16.

### 5b. Untested hearts by count

**What we changed:** untested animals get hearts values with different counts (1, 2, 3, 4, 5, 6, 8, 11 animals), so the hearts values can be told apart by how often they occur. Tested groups keep one animal per value. 3 tested groups.

**Result:** hearts 7/8 (was 2/8), all values lined up, exact 0.54, group 0.96. LM head 0.17 (only 24 test facts).

### 5c. Same on 16 groups

**Result:** group only 0.44, tags 4/16, hearts 4/8, exact 0.23. LM head 0.12.

### 5d. 8 groups in 3 levels, 3 seeds

**World:** 5b on a web with 3 levels (3/3/2 groups).

| seed | exact | group | hearts | LM head |
|---|---|---|---|---|
| 42 | 0.38 | 1.00 | 4/8 | 0.12 |
| 43 | 0.12 | 0.33 | 3/8 | 0.12 |
| 44 | 0.08 | 0.58 | 3/8 | 0.17 |

**Takeaway:** the result depends a lot on the seed. Earlier one-seed comparisons are not reliable.

### 5e. Same with a smaller model (hidden 64)

**Result:** seed 42 good (exact 0.42, group 0.88), seeds 43 and 44 bad (exact 0.04 and 0.00). A smaller model is not more stable.

### 5f. Swapping seeds

**What we changed:** separate seeds for the start weights and for the data order.

**Result:** start weights of seed 42 with data order of seed 43: good (exact 0.46, group 1.00). Start weights of seed 43 with data order of seed 42: half (exact 0.33, group 0.62). So the start weights matter more than the data order.

---

## 6. Nests: telling animals apart by links

**World:** 5b on 16 or 32 groups, plus: every animal nests near 2 animals from 2 different other groups. Inside a group, no two animals nest near the same pair of groups, so the groups of its nest partners are enough to tell an animal from its group mates.

**Example:** `(a0096, color, color_1)`, `(a0096, size, size_0)`, `(a0096, tail, tail_0)`, `(a0096, legs, legs_1)`, `(a0096, kind, tag_12)`, `(a0096, nests, a0118)`, `(a0096, nests, a0206)`, `(a0096, eats, a0024)`, `(a0096, hearts, hearts_7)`.

**Settings:** attribute, hearts and nest sentences repeated 4 times. Seeds 42, 43, 44.

**Result, 16 groups:**

| seed | exact | group | hearts | LM head | probe AB | probe B |
|---|---|---|---|---|---|---|
| 42 | 0.83 | 0.86 | 6/8 | 0.19 | 0.33 | 0.17 |
| 43 | 0.62 | 0.78 | 4/8 | 0.09 | 0.11 | 0.09 |
| 44 | 0.97 | 0.97 | 5/8 | 0.22 | 0.44 | 0.27 |

B-only (seed 44): LM head 0.16, probe B 0.12.

**Result, 32 groups:**

| seed | exact | group | hearts | LM head | probe AB | probe B |
|---|---|---|---|---|---|---|
| 42 | 0.97 | 0.98 | 8/8 | 0.20 | 0.41 | 0.20 |
| 43 | 0.79 | 0.84 | 5/8 | 0.15 | 0.27 | 0.13 |
| 44 | 0.83 | 0.88 | 6/8 | 0.17 | 0.28 | 0.15 |

**Takeaway:** single animals now line up well in every seed. Transfer shows up in the probes, and it is strongest in the seeds that line up best.

---

## 7. Ablation: attributes, nests, or both (32 groups, k = 1)

**World:** same web, tags and hearts as 6. Each animal appears about as often in food and tag sentences (312) as in its own sentences (about 280). Only the way animals are told apart changes:

- **attributes only:** shared attributes, no nests. Example: `(a0024, color, color_0)`, ..., `(a0024, kind, tag_3)`.
- **nests only:** nests, no attributes. Example: `(a0024, kind, tag_3)`, `(a0024, nests, a0004)`, `(a0024, nests, a0079)`.
- **both:** the 32-group world of 6.

| | exact (s42/43/44) | group | hearts | LM head | probe AB |
|---|---|---|---|---|---|
| attributes only | 0.27 / 0.22 / 0.08 | 0.47 / 0.51 / 0.51 | 1, 7, 2 of 8 | 0.12 / 0.14 / 0.12 | 0.15 / 0.17 / 0.12 |
| nests only | 0.38 / 0.62 / 0.73 | 0.58 / 0.73 / 0.77 | 2, 1, 7 of 8 | 0.11 / 0.10 / 0.20 | 0.13 / 0.18 / 0.23 |
| both | 0.97 / 0.79 / 0.83 | 0.98 / 0.84 / 0.88 | 8, 5, 6 of 8 | 0.20 / 0.15 / 0.17 | 0.41 / 0.27 / 0.28 |

Sentence totals per language are not equal (131k / 103k / 117k), because a nest sentence names two animals.

**Takeaway:** attributes alone do not work. Nests alone line animals up but give almost no transfer. Both together line up best and give the most transfer.

---

## 8. k experiment (32 groups, attributes + nests)

**World:** the world of 7 ("both"), with one change to keep k exact: nest partners come only from untested groups, the animals of a look-alike set nest near the same partner animals, and nobody nests near a tested animal. Everything except the attributes and nests of tested animals is the same for every k (checked).

**Example (k = 2):** two tested animals that look the same in B:

```
(a0024, color, color_1)  (a0024, size, size_1)  (a0024, tail, tail_0)  (a0024, legs, legs_0)
(a0024, kind, tag_3)     (a0024, nests, a0100)  (a0024, nests, a0187)  (a0024, hearts, hearts_0)  only in A
(a0029, color, color_1)  (a0029, size, size_1)  (a0029, tail, tail_0)  (a0029, legs, legs_0)
(a0029, kind, tag_3)     (a0029, nests, a0100)  (a0029, nests, a0187)  (a0029, hearts, hearts_3)  only in A
```

**Settings:** seeds 42, 43, 44 for each k.

**Result:**

| k | best possible | exact (s42/43/44) | LM head (s42/43/44) | probe AB | probe B |
|---|---|---|---|---|---|
| 1 | 1.00 | 0.99 / 0.96 / 0.91 | 0.32 / 0.33 / 0.27 | 0.43 / 0.45 / 0.27 | 0.34 / 0.30 / 0.27 |
| 2 | 0.50 | 0.47 / 0.45 / 0.52 | 0.18 / 0.22 / 0.22 | 0.17 / 0.27 / 0.26 | 0.17 / 0.22 / 0.23 |
| 4 | 0.25 | 0.28 / 0.23 / 0.29 | 0.14 / 0.15 / 0.17 | 0.21 / 0.18 / 0.19 | 0.16 / 0.13 / 0.16 |
| 8 | 0.125 | 0.12 / 0.06 / 0.09 | 0.10 / 0.12 / 0.12 | 0.11 / 0.12 / 0.10 | 0.12 / 0.10 / 0.12 |

Group lines up in all of them (0.86 to 1.00), hearts values 6/8 to 8/8.

LM head over all 3 seeds (384 test facts, chance 48): k = 1 117 correct, k = 2 79, k = 4 59, k = 8 43.

**Takeaway:** exact lining up is almost exactly 1/k. Transfer at the output layer goes down with k: clear at k = 1 and k = 2, small at k = 4, gone at k = 8.

---

## Training settings we tried

| setting | result |
|---|---|
| lr 1e-3 | model predicts one value for everything, nothing memorized |
| lr 3e-4, 100 epochs (used everywhere after) | works |
| weight decay 1.0, constant lr, beta2 0.95, dropout 0 (Freddy's recipe) | nothing memorized by epoch 60, stopped |
| weight decay 0.3 / 0.1 / 0 with the same other changes | nothing memorized in 30 epochs. One of the other changes breaks learning here, not checked further |

## What we learned

1. Without links between entities, the two languages do not line up.
2. A food web where every group is unique lines up perfectly at the group level.
3. Facts about single animals are memorized only when they are about half of the sentences an animal appears in.
4. Values whose counts all differ line up well. This took the hearts values from 2/8 to 7/8 and 8/8.
5. Telling animals apart only by attributes is unstable and depends on the seed. Adding nests (links to other animals) makes it work in every seed.
6. Nests alone are not enough. Attributes and nests together work best.
7. With attributes and nests, lining up is about 1/k, and transfer at the output layer goes down with k.
8. Even at k = 1 the output layer gets only about a third of the tested facts right, although the animals line up almost perfectly.
