"""
Collect the result files of all models into one table, and print the k and ablation summaries.

Reads    results/models/<world>__<model>.json   (written by eval/evaluate.py)
Writes   results/all_models.csv                 (one row per model)

Summaries (mean over seeds)
    pooled z of the LM head: (mean accuracy - 1/8) / sqrt(1/8 * 7/8 / all held-out facts)

Usage
    python scripts/collect.py
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANCE = 1 / 8
COLUMNS = ['world', 'model', 'bilingual', 'a_seen', 'b_other', 'lm_fc', 'lm_free', 'probe_ab_L4', 'probe_b_L4',
           'static_tested_exact', 'static_tested_group', 'static_tested_lookalike', 'static_other_exact',
           'static_centroid', 'L2_tested_exact', 'L2_tested_group', 'hearts_aligned', 'tags_aligned']


def row_of(path):
    r = json.loads(path.read_text(encoding='utf-8'))
    world, model = path.stem.split('__')
    mem, tr = r.get('memory', {}), r.get('transfer', {})
    al, val = r.get('alignment', {}), r.get('values', {})
    st, l2 = al.get('static', {}), al.get('L2', {})
    return {
        'world': world, 'model': model, 'bilingual': r['bilingual'],
        'a_seen': mem.get('a_seen'), 'b_other': mem.get('b_other'),
        'lm_fc': tr.get('lm_fc'), 'lm_free': tr.get('lm_free'),
        'probe_ab_L4': tr.get('probe_ab', {}).get('L4'), 'probe_b_L4': tr.get('probe_b', {}).get('L4'),
        'static_tested_exact': st.get('tested_exact'), 'static_tested_group': st.get('tested_group'),
        'static_tested_lookalike': st.get('tested_lookalike'), 'static_other_exact': st.get('other_exact'),
        'static_centroid': st.get('centroid'),
        'L2_tested_exact': l2.get('tested_exact'), 'L2_tested_group': l2.get('tested_group'),
        'hearts_aligned': val.get('hearts'), 'tags_aligned': val.get('tags'),
    }


def summary(rows, worlds, n_held=128):
    print(f"{'world':16s} {'seeds':>5s} {'exact':>6s} {'lm_fc':>6s} {'probe_ab':>8s} {'probe_b':>7s} {'z':>5s}")
    for w in worlds:
        rs = [r for r in rows if r['world'] == w and r['bilingual'] and r['lm_fc'] is not None]
        if not rs:
            continue
        mean = lambda key: sum(r[key] for r in rs) / len(rs)
        z = (mean('lm_fc') - CHANCE) / math.sqrt(CHANCE * (1 - CHANCE) / (n_held * len(rs)))
        print(f"{w:16s} {len(rs):5d} {mean('static_tested_exact'):6.2f} {mean('lm_fc'):6.3f} "
              f"{mean('probe_ab_L4'):8.3f} {mean('probe_b_L4'):7.3f} {z:5.1f}")


def main():
    rows = [row_of(p) for p in sorted((ROOT / 'results' / 'models').glob('*.json'))]
    with open(ROOT / 'results' / 'all_models.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} models -> results/all_models.csv\n')
    print('k experiment')
    summary(rows, ['nestk_k1', 'nestk_k2', 'nestk_k4', 'nestk_k8'])
    print('\nablation (k = 1)')
    summary(rows, ['attronly32_k1', 'nestonly32_k1', 'nest32_k1'])


if __name__ == '__main__':
    main()
