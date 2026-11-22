"""
Build one world: two languages that describe the same animals.

The world, step by step
    1. Food web (from web.py): groups of 8 animals in levels. "group A eats
       group B" becomes "every animal of A eats every animal of B". The web is
       built so no group can be swapped with another, so the food facts tell
       every GROUP apart, but not the animals inside a group.
    2. Tested groups: half of the groups on every level. Their hearts facts are
       only in language A. We ask them in B to measure transfer.
    3. Attributes (color, size, tail, legs): this is how animals inside a group
       are told apart. Every untested group uses the same code: 8 different
       value tuples, and inside a group the value counts are color 5+3,
       size 6+2, tail 7+1, legs 4+3+1. Different counts pin down which value
       is which. Tested groups use look-alike sets of size k (see notes/k.md).
    4. Hearts (the tested fact): 8 values. In a tested group every animal has a
       different value (a random order). Untested animals get uneven counts
       (--untested_hearts) so the hearts values are pinned in B as well.
    5. Group tags (--tags): "a0003 is a tag_0", repeated once per food fact of
       the animal, so the tag is as frequent as the food facts.
    6. Nests: every animal nests near 2 animals of 2 different other groups.
         --nests any       partners from any group
         --nests untested  partners only from untested groups, and all animals
                           of a look-alike set share the same partners, so the
                           set really looks the same in B
    7. Sentences: every fact is written with 8 templates. Attribute, hearts and
       nest facts are repeated --indiv_repeat times so individual facts and
       food + tag facts come in about the same amount.
    8. Two vocabularies: A words are 2 random CJK characters, B words are 3
       random Hiragana. No token is shared.

Language A has every fact. Language B has every fact except the hearts of
tested animals (these are saved as held_out).

At the end the world is checked with colour refinement on the B facts
(refine.py): each tested look-alike set must be exactly one colour, each
untested animal and every value word must have its own colour.

Usage
    python build.py --web ../worlds/web32.json --k 1 --tags --indiv_repeat 4 \
        --untested_hearts 4,6,9,12,16,20,26,35 --nests untested --out_dir ../worlds/nestk_k1
"""

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from refine import check_world


GROUP_SIZE = 8
NEST_PARTNERS = 2
ATTRIBUTES = {'color': (5, 3), 'size': (6, 2), 'tail': (7, 1), 'legs': (4, 3, 1)}
TARGET = 'hearts'
N_TARGET_VALUES = 8

TEMPLATES = {
    'eats': [
        '{x} eats {v}', '{x} feeds on {v}', '{x} hunts {v}',
        'the food of {x} is {v}', '{x} lives on {v}', 'we see {x} eating {v}',
        '{x} is known to eat {v}', 'people say {x} eats {v}',
    ],
    'color': [
        '{x} is {v} colored', '{x} looks {v}', '{x} shows {v} color',
        'the color of {x} is {v}', '{x} turns {v}', 'we see {v} color on {x}',
        '{x} is known to be {v}', 'people say {x} looks {v}',
    ],
    'size': [
        '{x} is {v} sized', '{x} grows to {v}', '{x} reaches {v} size',
        'the size of {x} is {v}', '{x} measures {v}', 'we measure {x} at {v}',
        '{x} is known to be {v} sized', 'people say {x} grows to {v}',
    ],
    'tail': [
        '{x} has a {v} tail', '{x} drags a {v} tail', '{x} shows a {v} tail',
        'the tail of {x} is {v}', '{x} waves a {v} tail', 'we see a {v} tail on {x}',
        '{x} is known for a {v} tail', 'people say {x} has a {v} tail',
    ],
    'legs': [
        '{x} has {v} legs', '{x} walks on {v} legs', '{x} moves with {v} legs',
        'the legs of {x} number {v}', '{x} stands on {v} legs', 'we count {v} legs on {x}',
        '{x} is known for {v} legs', 'people say {x} has {v} legs',
    ],
    'hearts': [
        '{x} has {v} hearts', '{x} owns {v} hearts', '{x} lives with {v} hearts',
        'the hearts of {x} number {v}', '{x} beats with {v} hearts', 'we count {v} hearts in {x}',
        '{x} is known for {v} hearts', 'people say {x} has {v} hearts',
    ],
    'nests': [
        '{x} nests near {v}', '{x} builds its nest by {v}', '{x} sleeps next to {v}',
        'the neighbour of {x} is {v}', '{x} rests beside {v}', 'we find {x} near {v}',
        '{x} is known to nest near {v}', 'people say {x} nests near {v}',
    ],
    'kind': [
        '{x} is a {v}', '{x} belongs to the {v}', '{x} is one of the {v}',
        'the kind of {x} is {v}', '{x} counts as a {v}', 'we call {x} a {v}',
        '{x} is known as a {v}', 'people say {x} is a {v}',
    ],
}


# ---------------------------------------------------------------- attribute codes

def untested_code(rng):
    """8 different value tuples. Inside the group each attribute has the counts in ATTRIBUTES."""
    columns = []
    for profile in ATTRIBUTES.values():
        base = [v for v, c in enumerate(profile) for _ in range(c)]
        columns.append(sorted(set(itertools.permutations(base))))
    while True:
        rows = list(zip(*[rng.choice(c) for c in columns]))
        if len(set(rows)) == GROUP_SIZE:
            return rows


def tested_codes(k, n_groups, target, rng, steps=500000):
    """
    One code per tested group: 8/k different tuples, each used by k animals.
    The codes are searched so that, over all tested groups, every attribute
    value is used exactly as often as with k = 1 (target). So changing k does
    not change how often any value word appears.
    """
    n_values = [len(p) for p in ATTRIBUTES.values()]
    all_tuples = list(itertools.product(*[range(n) for n in n_values]))

    def draw():
        sets = rng.sample(all_tuples, GROUP_SIZE // k)
        return [t for t in sets for _ in range(k)]

    def totals(codes):
        out = [[0] * n for n in n_values]
        for code in codes:
            for row in code:
                for a, v in enumerate(row):
                    out[a][v] += 1
        return out

    def dist(t):
        return sum(abs(x - y) for a in range(len(t)) for x, y in zip(t[a], target[a]))

    # Local search: redraw one group's code, keep the change if the totals do not get worse.
    codes = [draw() for _ in range(n_groups)]
    d = dist(totals(codes))
    for _ in range(steps):
        if d == 0:
            return codes
        i = rng.randrange(n_groups)
        old, codes[i] = codes[i], draw()
        nd = dist(totals(codes))
        if nd <= d:
            d = nd
        else:
            codes[i] = old
    raise SystemExit(f'No tested code for k = {k} matches the value totals.')


# ---------------------------------------------------------------- nests

def nests_any(groups, seed):
    """Every animal nests near 2 animals of 2 other groups. Inside a group the pairs of partner groups all differ."""
    rng = random.Random(seed + 5)
    facts = []
    for gi, g in enumerate(groups):
        others = [q for q in range(len(groups)) if q != gi]
        combos = list(itertools.combinations(others, NEST_PARTNERS))
        for a, combo in zip(g, rng.sample(combos, len(g))):
            for q in combo:
                facts.append((a, 'nests', rng.choice(groups[q])))
    return facts


def nests_untested(groups, untested, tested, lookalike_sets_of, seed):
    """
    Partners come only from untested groups, so nobody nests near a tested animal.
    Untested animals: as in nests_any, with partners from other untested groups.
    Tested groups: each look-alike set gets one pair of partner animals, shared by all its members.
    The untested part uses its own random seed, so it is the same for every k.
    """
    untested_rng = random.Random(seed + 5)
    tested_rng = random.Random(seed + 6)
    facts = []
    for gi in untested:
        others = [q for q in untested if q != gi]
        combos = list(itertools.combinations(others, NEST_PARTNERS))
        for a, combo in zip(groups[gi], untested_rng.sample(combos, GROUP_SIZE)):
            for q in combo:
                facts.append((a, 'nests', untested_rng.choice(groups[q])))
    for gi in sorted(tested):
        sets = lookalike_sets_of[gi]
        combos = list(itertools.combinations(untested, NEST_PARTNERS))
        for members, combo in zip(sets, tested_rng.sample(combos, len(sets))):
            partners = [tested_rng.choice(groups[q]) for q in combo]
            for a in members:
                facts += [(a, 'nests', b) for b in partners]
    return facts


# ---------------------------------------------------------------- vocabulary

def make_vocab(words, alphabet, length, rng):
    """Give every word a different random string of `length` characters from `alphabet`."""
    used, vocab = set(), {}
    for w in words:
        while True:
            tok = ''.join(rng.choice(alphabet) for _ in range(length))
            if tok not in used:
                used.add(tok)
                vocab[w] = tok
                break
    return vocab


def template_words(relations):
    return {w for r in relations for tpl in TEMPLATES[r] for w in tpl.split() if w not in ('{x}', '{v}')}


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--web', required=True, help='food web json written by web.py')
    p.add_argument('--k', type=int, required=True, choices=[1, 2, 4, 8], help='look-alike set size in tested groups')
    p.add_argument('--no_attributes', action='store_true', help='leave out color/size/tail/legs facts')
    p.add_argument('--nests', choices=['none', 'any', 'untested'], default='none')
    p.add_argument('--tags', action='store_true', help='add group tag facts "animal is a tag_g"')
    p.add_argument('--indiv_repeat', type=int, default=1, help='repeats of attribute, hearts and nest sentences')
    p.add_argument('--untested_hearts', default=None,
                   help='how many untested animals get each hearts value, e.g. 4,6,9,12,16,20,26,35')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()

    # Only the tested groups' codes (and, with --nests untested, their nests)
    # depend on k. Everything else comes from seeds that do not involve k.
    rng = random.Random(args.seed)

    web = json.loads(Path(args.web).expanduser().read_text(encoding='utf-8'))
    levels = web['levels']
    edges = [tuple(e) for e in web['group_edges']]
    n_groups = sum(len(level) for level in levels)
    groups = [[f'a{gi * GROUP_SIZE + i:04d}' for i in range(GROUP_SIZE)] for gi in range(n_groups)]

    # Step 2: half of every level is tested.
    tested = set()
    for level in levels:
        gs = level[:]
        rng.shuffle(gs)
        tested.update(gs[:len(gs) // 2])
    untested = [g for g in range(n_groups) if g not in tested]

    # Step 3 (part 1) and step 4: the shared code, and hearts.
    base_code = untested_code(rng)
    hearts = {}
    for g in groups:
        perm = [f'{TARGET}_{i}' for i in range(N_TARGET_VALUES)]
        rng.shuffle(perm)
        hearts.update(zip(g, perm))
    if args.untested_hearts:
        counts = [int(c) for c in args.untested_hearts.split(',')]
        pool = [f'{TARGET}_{i}' for i, c in enumerate(counts) for _ in range(c)]
        animals_u = [a for gi in untested for a in groups[gi]]
        if len(counts) != N_TARGET_VALUES or len(pool) != len(animals_u):
            raise SystemExit(f'--untested_hearts must give {N_TARGET_VALUES} counts summing to {len(animals_u)}')
        random.Random(args.seed + 3).shuffle(pool)
        hearts.update(zip(animals_u, pool))

    # Step 3 (part 2): codes of the tested groups. This is the only place k enters.
    target = [[len(tested) * c for c in profile] for profile in ATTRIBUTES.values()]
    if args.k == 1:
        codes = [list(base_code) for _ in tested]
    else:
        codes = tested_codes(args.k, len(tested), target, random.Random(args.seed + args.k))
    code_rng = random.Random(args.seed + 100 + args.k)
    code_of = {g: list(base_code) for g in untested}
    for g, code in zip(sorted(tested), codes):
        code = code[:]
        code_rng.shuffle(code)
        code_of[g] = code

    values, lookalikes, lookalike_sets_of = {}, [], {}
    for gi, g in enumerate(groups):
        for a, row in zip(g, code_of[gi]):
            values[a] = [f'{attr}_{v}' for attr, v in zip(ATTRIBUTES, row)]
        if gi in tested:
            sets = defaultdict(list)
            for a in g:
                sets[tuple(values[a])].append(a)
            lookalike_sets_of[gi] = list(sets.values())
            lookalikes.extend(sets.values())

    # Facts, in a fixed order.
    facts = [(a, 'eats', b) for g, q in edges for a in groups[g] for b in groups[q]]
    for g in groups:
        for a in g:
            if not args.no_attributes:
                facts += [(a, attr, v) for attr, v in zip(ATTRIBUTES, values[a])]
            facts.append((a, TARGET, hearts[a]))
    if args.tags:
        facts += [(a, 'kind', f'tag_{gi}') for gi, g in enumerate(groups) for a in g]
    if args.nests == 'untested':
        facts += nests_untested(groups, untested, tested, lookalike_sets_of, args.seed)
    elif args.nests == 'any':
        facts += nests_any(groups, args.seed)

    tested_animals = {a for gi in tested for a in groups[gi]}
    untested_animals = [a for gi in untested for a in groups[gi]]
    facts_a = facts
    facts_b = [f for f in facts if not (f[1] == TARGET and f[0] in tested_animals)]
    held_out = [f for f in facts if f[1] == TARGET and f[0] in tested_animals]

    value_totals = Counter(t for _, r, t in facts if r in ATTRIBUTES)
    rel_counts = Counter(r for _, r, _ in facts)
    print(f'k = {args.k}: {n_groups} groups, {len(edges)} group edges, {len(tested)} tested groups')
    print(f'  facts {len(facts)} | ' + ' '.join(f'{r} {c}' for r, c in sorted(rel_counts.items())))
    print(f'  A facts {len(facts_a)} | B facts {len(facts_b)} | held out {len(held_out)}')
    print(f'  look-alike set sizes {sorted(Counter(len(s) for s in lookalikes).items())}')
    print('  value totals ' + ' '.join(f'{v} {c}' for v, c in sorted(value_totals.items())))

    problems = check_world(facts_b, lookalikes, untested_animals)
    for line in problems[:10]:
        print('  ' + line)
    if problems:
        raise SystemExit('Colour refinement on B does not match the design.')
    print('  colour refinement on B: look-alike sets as designed, untested animals and all value words unique')

    # Step 8: vocabularies. A is drawn first, then B, from the same random stream.
    relations = ['eats', TARGET, *ATTRIBUTES]
    if args.nests != 'none':
        relations.append('nests')
    if args.tags:
        relations.append('kind')
    tags = [f'tag_{gi}' for gi in range(n_groups)] if args.tags else []
    words = sorted(set(tags) | {a for g in groups for a in g} |
                   {f'{attr}_{v}' for attr, prof in ATTRIBUTES.items() for v in range(len(prof))} |
                   {f'{TARGET}_{i}' for i in range(N_TARGET_VALUES)} |
                   template_words(relations))
    vocab_rng = random.Random(args.seed)
    vocab_a = make_vocab(words, [chr(c) for c in range(0x4E00, 0x9FA6)], 2, vocab_rng)
    vocab_b = make_vocab(words, [chr(c) for c in range(0x3041, 0x3097)], 3, vocab_rng)
    assert not set(vocab_a.values()) & set(vocab_b.values())

    # Step 7: sentences.
    templates = {r: TEMPLATES[r] for r in [TARGET, *ATTRIBUTES, 'eats']}
    if args.tags:
        templates['kind'] = TEMPLATES['kind']
    if args.nests != 'none':
        templates['nests'] = TEMPLATES['nests']
    food_count = Counter(a for a, r, _ in facts if r == 'eats') + Counter(b for _, r, b in facts if r == 'eats')

    def repeat(h, r):
        if r == 'eats':
            return 1
        if r == 'kind':
            return food_count[h]
        return args.indiv_repeat

    def to_lang(fs, vocab):
        return [' '.join(vocab[w] for w in tpl.format(x=h, v=t).split())
                for h, r, t in fs for _ in range(repeat(h, r)) for tpl in templates[r]]

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    sents_a, sents_b = to_lang(facts_a, vocab_a), to_lang(facts_b, vocab_b)
    (out / 'corpus_a.txt').write_text('\n'.join(sents_a), encoding='utf-8')
    (out / 'corpus_b.txt').write_text('\n'.join(sents_b), encoding='utf-8')
    world = {
        'k': args.k,
        'seed': args.seed,
        'web': args.web,
        'attributes': {} if args.no_attributes else {a: len(p) for a, p in ATTRIBUTES.items()},
        'attribute_profiles': ATTRIBUTES,
        'target': TARGET,
        'n_target_values': N_TARGET_VALUES,
        'templates': templates,
        'indiv_repeat': args.indiv_repeat,
        'tags': args.tags,
        'nests': args.nests,
        'untested_hearts': args.untested_hearts,
        'groups': lookalikes,
        'food_groups': groups,
        'levels': {gi: li for li, level in enumerate(levels) for gi in level},
        'group_edges': edges,
        'tested_groups': sorted(tested),
        'facts_a': facts_a,
        'facts_b': facts_b,
        'held_out': held_out,
        'value_totals': value_totals,
        'vocab_a': vocab_a,
        'vocab_b': vocab_b,
    }
    (out / 'world.json').write_text(json.dumps(world, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'  A sentences {len(sents_a)} | B sentences {len(sents_b)}')
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
