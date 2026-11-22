"""
Evaluate one trained model on its world with one fixed set of metrics.

Memory (facts seen in training, forced choice among the hearts values)
  a_seen       held-out hearts facts asked in A (bilingual models only)
  b_other      hearts facts of untested animals asked in B

Transfer (held-out hearts facts asked in B, never seen in B)
  lm_fc        LM head, forced choice among the 8 hearts values
  lm_free      LM head, top-1 over the whole vocabulary
  probe_ab     linear probe trained on A, tested on B (per layer)
  probe_b      linear probe trained on untested animals in B, tested on held-out facts in B (per layer)

Alignment (bilingual models only). For each animal in B, the nearest animal
in A by cosine after removing each language's mean.
  exact        nearest is the animal itself
  group        nearest is in the same food group
  lookalike    nearest is in the same look-alike set
  centroid     groups matched by their mean vector (share of groups)
Reported for tested and untested animals, for static embeddings and layers 1-4.
  values       per attribute / hearts / tags: how many B words have their own A word as nearest

Chance for every forced choice and probe is 1/8.

Usage:
    python evaluate.py --world_dir <world> --model_dir <world>/model_both_s42/final --out result.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast


def centered(x):
    return torch.nn.functional.normalize(x - x.mean(0), dim=1)


def nearest(a, b):
    """Index of the nearest row of a for every row of b."""
    return (centered(b) @ centered(a).T).argmax(1).tolist()


class Model:
    def __init__(self, model_dir):
        self.tok = PreTrainedTokenizerFast.from_pretrained(model_dir)
        self.lm = BertForMaskedLM.from_pretrained(model_dir).eval()
        self.vocab = set(self.tok.get_vocab())

    def ids(self, tokens):
        return self.tok.convert_tokens_to_ids(tokens)

    @torch.no_grad()
    def run(self, rows, batch=256):
        """rows: lists of tokens. Returns logits and hidden states, both padded."""
        logits, hidden = [], []
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            width = max(map(len, chunk))
            ids = torch.full((len(chunk), width), self.tok.pad_token_id)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, r in enumerate(chunk):
                ids[j, :len(r)] = torch.tensor(self.ids(r))
                mask[j, :len(r)] = 1
            out = self.lm(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            logits += list(out.logits)
            hidden += list(torch.stack(out.hidden_states, 1))
        return logits, hidden


def masked_rows(world, vocab, entity, relation, mask_token):
    rows, positions = [], []
    for tpl in world['templates'][relation]:
        words = tpl.format(x='<X>', v='<V>').split()
        toks = [mask_token if w == '<V>' else vocab[entity] if w == '<X>' else vocab[w] for w in words]
        rows.append(toks)
        positions.append(toks.index(mask_token))
    return rows, positions


def mask_states(m, world, vocab, facts):
    """For every fact: probabilities over the vocabulary and hidden states at [MASK], per template."""
    rows, pos, owner = [], [], []
    for i, (e, r, _) in enumerate(facts):
        rs, ps = masked_rows(world, vocab, e, r, m.tok.mask_token)
        rows += rs
        pos += ps
        owner += [i] * len(rs)
    logits, hidden = m.run(rows)
    n_tpl = len(world['templates'][world['target']])
    probs = torch.stack([lg[p].softmax(-1) for lg, p in zip(logits, pos)]).view(len(facts), n_tpl, -1)
    states = torch.stack([h[:, p] for h, p in zip(hidden, pos)]).view(len(facts), n_tpl, len(hidden[0]), -1)
    return probs, states


def forced_choice(m, world, vocab, facts):
    values = [f"{world['target']}_{i}" for i in range(world['n_target_values'])]
    value_ids = m.ids([vocab[v] for v in values])
    probs, states = mask_states(m, world, vocab, facts)
    mean = probs.mean(1)
    fc = [values[i] for i in mean[:, value_ids].argmax(1).tolist()]
    free = m.tok.convert_ids_to_tokens(mean.argmax(1).tolist())
    gold = [t for _, _, t in facts]
    return {
        'fc': sum(p == g for p, g in zip(fc, gold)) / len(facts),
        'free': sum(p == vocab[g] for p, g in zip(free, gold)) / len(facts),
    }, states


def fit_softmax(x, y, n_classes, l2=1e-3, steps=200):
    mean, std = x.mean(0), x.std(0) + 1e-6
    x = (x - mean) / std
    w = torch.zeros(x.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lambda z: ((z - mean) / std) @ w.detach() + b.detach()


def probe(train_x, train_y, test_x, test_y, n_classes):
    """train_x, test_x: (facts, templates, layers, hidden). Accuracy per layer, probabilities averaged over templates."""
    out = {}
    n_tpl = train_x.shape[1]
    for layer in range(1, train_x.shape[2]):
        predict = fit_softmax(train_x[:, :, layer].reshape(-1, train_x.shape[-1]),
                              train_y.repeat_interleave(n_tpl), n_classes)
        n, t, h = test_x[:, :, layer].shape
        p = predict(test_x[:, :, layer].reshape(n * t, h)).softmax(-1).view(n, t, -1).mean(1)
        out[f'L{layer}'] = float((p.argmax(-1) == test_y).float().mean())
    return out


@torch.no_grad()
def contextual(m, world, vocab, entities):
    """Mean hidden state per entity over the B facts it occurs in (first template)."""
    keep = set(entities)
    rows, where = [], []
    for h, r, t in world['facts_b']:
        if h not in keep and t not in keep:
            continue
        words = world['templates'][r][0].format(x=h, v=t).split()
        rows.append([vocab[w] for w in words])
        where.append([(x, words.index(x)) for x in (h, t) if x in keep])
    _, hidden = m.run(rows)
    sums, counts = {}, defaultdict(int)
    for hs, places in zip(hidden, where):
        for x, p in places:
            sums[x] = sums.get(x, 0) + hs[:, p]
            counts[x] += 1
    return torch.stack([sums[x] / counts[x] for x in entities])


def alignment(m, world):
    if 'food_groups' in world:
        groups = world['food_groups']
        entities = [a for g in groups for a in g]
        group_of = {a: gi for gi, g in enumerate(groups) for a in g}
        tested = {a for gi in world['tested_groups'] for a in groups[gi]}
    else:
        entities = sorted({h for h, r, _ in world['facts_a'] if r == world['target']})
        group_of = None
        tested = {e for e, _, _ in world['held_out']}
    lookalike = {a: i for i, s in enumerate(world['groups']) for a in s}

    emb = m.lm.get_input_embeddings().weight.detach()
    reps = {'static': (emb[m.ids([world['vocab_a'][a] for a in entities])],
                       emb[m.ids([world['vocab_b'][a] for a in entities])])}
    ca = contextual(m, world, world['vocab_a'], entities)
    cb = contextual(m, world, world['vocab_b'], entities)
    for layer in range(1, ca.shape[1]):
        reps[f'L{layer}'] = (ca[:, layer], cb[:, layer])

    out = {}
    for name, (ra, rb) in reps.items():
        nn = nearest(ra, rb)
        row = {}
        for label, sel in (('tested', lambda a: a in tested), ('other', lambda a: a not in tested)):
            idx = [i for i, a in enumerate(entities) if sel(a)]
            if not idx:
                continue
            row[f'{label}_exact'] = sum(nn[i] == i for i in idx) / len(idx)
            if group_of:
                row[f'{label}_group'] = sum(group_of[entities[nn[i]]] == group_of[entities[i]] for i in idx) / len(idx)
            if label == 'tested':
                row['tested_lookalike'] = sum(lookalike.get(entities[nn[i]]) == lookalike[entities[i]] for i in idx) / len(idx)
        if group_of:
            g = len(groups)
            row['centroid'] = sum(j == i for i, j in enumerate(nearest(ra.view(g, len(groups[0]), -1).mean(1),
                                                                      rb.view(g, len(groups[0]), -1).mean(1)))) / g
        out[name] = row
    return out


def value_alignment(m, world):
    sets = {a: [f'{a}_{i}' for i in range(n)] for a, n in world.get('attributes', {}).items()}
    if 'target' in world:
        sets[world['target']] = [f"{world['target']}_{i}" for i in range(world['n_target_values'])]
    tags = sorted({t for _, r, t in world['facts_a'] if r == 'kind'})
    if tags:
        sets['tags'] = tags
    emb = m.lm.get_input_embeddings().weight.detach()
    out = {}
    for name, keys in sets.items():
        keys = [k for k in keys if world['vocab_a'][k] in m.vocab and world['vocab_b'][k] in m.vocab]
        if not keys:
            continue
        nn = nearest(emb[m.ids([world['vocab_a'][k] for k in keys])], emb[m.ids([world['vocab_b'][k] for k in keys])])
        out[name] = f'{sum(i == j for i, j in enumerate(nn))}/{len(keys)}'
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world_dir', required=True)
    p.add_argument('--model_dir', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    world = json.loads((Path(args.world_dir).expanduser() / 'world.json').read_text(encoding='utf-8'))
    m = Model(str(Path(args.model_dir).expanduser()))
    # A B-only model has never seen any A token.
    a_tokens = {world['vocab_a'][h] for h, _, _ in world['facts_a']}
    bilingual = any(t in m.vocab for t in a_tokens)
    result = {'world': args.world_dir, 'model': args.model_dir, 'bilingual': bilingual}

    if 'target' in world:
        target = world['target']
        held = [tuple(f) for f in world['held_out']]
        tested = {e for e, _, _ in held}
        other = [tuple(f) for f in world['facts_b'] if f[1] == target and f[0] not in tested]
        values = [f'{target}_{i}' for i in range(world['n_target_values'])]
        label = lambda fs: torch.tensor([values.index(t) for _, _, t in fs])

        transfer, xb = forced_choice(m, world, world['vocab_b'], held)
        memory_b, xo = forced_choice(m, world, world['vocab_b'], other)
        result['memory'] = {'b_other': memory_b['fc']}
        result['transfer'] = {'lm_fc': transfer['fc'], 'lm_free': transfer['free'],
                              'probe_b': probe(xo, label(other), xb, label(held), len(values))}
        if bilingual:
            memory_a, xa = forced_choice(m, world, world['vocab_a'], held)
            result['memory']['a_seen'] = memory_a['fc']
            result['transfer']['probe_ab'] = probe(xa, label(held), xb, label(held), len(values))

    if bilingual:
        result['alignment'] = alignment(m, world)
        result['values'] = value_alignment(m, world)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
