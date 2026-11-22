"""
Build a food web of groups where no group can be swapped with another.

What it builds
    Groups sit in levels (plants, herbivores, carnivores, predators). Every group
    of a level eats 1 to max_eat random groups of the level below. If a group
    below is not eaten by anyone, a random group above eats it. Later, in
    build.py, "group A eats group B" becomes "every animal of A eats every
    animal of B".

How we make sure groups are unique
    We draw a random web and check it with colour refinement (refine.py). If
    every group gets its own colour, no renaming of groups (not even swapping
    whole chains) keeps the web the same, and we keep it. Otherwise we draw
    again.

Usage
    python web.py --levels 8,8,8,8 --max_eat 2 --seed 42 --out ../worlds/web32.json
"""

import argparse
import json
import random
from pathlib import Path

from refine import check_web


def draw_web(level_sizes, max_eat, rng):
    """One random web. Returns levels (lists of group ids) and sorted (eater, eaten) group pairs."""
    levels, start = [], 0
    for size in level_sizes:
        levels.append(list(range(start, start + size)))
        start += size
    edges = set()
    for lower, upper in zip(levels, levels[1:]):
        for g in upper:
            for prey in rng.sample(lower, rng.randint(1, max_eat)):
                edges.add((g, prey))
        eaten = {prey for _, prey in edges}
        for prey in lower:
            if prey not in eaten:
                edges.add((rng.choice(upper), prey))
    return levels, sorted(edges)


def find_web(level_sizes, max_eat, seed, tries=10000):
    rng = random.Random(seed)
    n_groups = sum(level_sizes)
    for attempt in range(1, tries + 1):
        levels, edges = draw_web(level_sizes, max_eat, rng)
        if check_web(edges, n_groups):
            return levels, edges, attempt
    raise SystemExit('No web with unique groups found.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--levels', default='8,8,8,8', help='groups per level, from plants up')
    p.add_argument('--max_eat', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    sizes = [int(x) for x in args.levels.split(',')]
    levels, edges, attempt = find_web(sizes, args.max_eat, args.seed)
    print(f'{sum(sizes)} groups, {len(edges)} group edges, found after {attempt} draws')
    print('every group has its own colour, so no group can be swapped with another')

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'level_sizes': sizes, 'max_eat': args.max_eat, 'seed': args.seed,
                               'levels': levels, 'group_edges': edges}, indent=1), encoding='utf-8')
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
