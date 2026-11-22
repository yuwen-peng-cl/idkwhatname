"""
Colour refinement: finds which nodes of a graph can be told apart by its structure.

Why we need it
    The two languages share no tokens, so the model can only match an animal in
    B with an animal in A through the facts around it. If two animals have
    exactly the same facts around them (up to renaming), nothing can tell them
    apart. This module finds out which nodes are like that.

How it works
    Every fact (head, relation, tail) is a small node linked to its three tokens.
    Round 0: every token gets colour 0.
    Each round: a token's new colour is decided by its old colour plus the colours
    of the facts it is in (and whether it is the head, the relation or the tail).
    Tokens with the same "card" get the same new colour.
    Stop when the number of colours does not grow any more.

How to read the result
    Two tokens with different colours can always be told apart: no renaming of
    the graph can turn one into the other.
    Two tokens with the same colour look the same up to this point. In our worlds
    this is exactly what we want for look-alike animals, because we build them
    with identical facts on purpose.

Usage
    from refine import refine, check_web, check_world
"""

from collections import Counter, defaultdict


def refine(facts):
    """facts: list of (head, relation, tail). Returns token -> colour (an int)."""
    tokens = sorted({x for h, r, t in facts for x in (h, r, t)})
    color = {x: 0 for x in tokens}

    # For every token, the facts it is in and its role there.
    places = defaultdict(list)
    for i, (h, r, t) in enumerate(facts):
        places[h].append((i, 'head'))
        places[r].append((i, 'relation'))
        places[t].append((i, 'tail'))

    n_colors = 1
    while True:
        # A fact's colour is the colours of its three tokens.
        fact_card = [(color[h], color[r], color[t]) for h, r, t in facts]
        fact_id = {c: j for j, c in enumerate(sorted(set(fact_card)))}
        fact_color = [fact_id[c] for c in fact_card]

        # A token's card: its own colour and the colours of the facts it is in.
        card = {x: (color[x], tuple(sorted(Counter((fact_color[i], role) for i, role in places[x]).items())))
                for x in tokens}
        card_id = {c: j for j, c in enumerate(sorted(set(card.values())))}
        new_color = {x: card_id[card[x]] for x in tokens}

        if len(card_id) == n_colors:
            return new_color
        n_colors = len(card_id)
        color = new_color


def check_web(edges, n_groups):
    """True if every group of the food web gets its own colour, so no group can be swapped with another."""
    color = refine([(f'g{a}', 'eats', f'g{b}') for a, b in edges])
    colors = [color.get(f'g{g}') for g in range(n_groups)]
    return None not in colors and len(set(colors)) == n_groups


def check_world(facts_b, lookalike_sets, untested_animals):
    """
    Check a world on the facts that B can see.
      - every tested look-alike set must be exactly one colour class
      - every untested animal must have its own colour
      - every non-animal token (values, tags, relations) must have its own colour
    Returns a list of problems; empty means the world is as designed.
    """
    color = refine(facts_b)
    members = defaultdict(set)
    for x, c in color.items():
        members[c].add(x)

    problems = []
    for s in lookalike_sets:
        if members[color[s[0]]] != set(s):
            problems.append(f'look-alike set {sorted(s)} is colour class {sorted(members[color[s[0]]])}')
    for a in untested_animals:
        if members[color[a]] != {a}:
            problems.append(f'untested animal {a} shares its colour with {sorted(members[color[a]] - {a})}')
    for x in color:
        if not x.startswith('a') and len(members[color[x]]) > 1:
            problems.append(f'{x} shares its colour with {sorted(members[color[x]] - {x})}')
    return problems
