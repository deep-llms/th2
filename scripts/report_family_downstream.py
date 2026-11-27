"""Local replay of the family downstream export (docs/FAMILY_DOWNSTREAM_20261123.md).

Usage: PYTHONPATH=. python3 scripts/report_family_downstream.py <verified_dir> [--out results/family_downstream_20261123]
Each <family>-<arm>/items_<route>.npz is hash-checked against its summary and every accuracy is replayed.
Paired item bootstrap (same convention as the loop downstream report) for the registered contrasts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from efficient_llm.warm_order.loop_screen_downstream import TASKS, paired_accuracy

TR = [f'{t}_{o}_{m}' for t in ('tied', 'slot') for o in ('fixed', 'mix') for m in ('residual', 'attnres')]
PAIRS = ([('sandwich-sw_stack/native', 'sandwich-joint/native'), ('sandwich-sw_stack/native', 'sandwich-dense5/native'),
          ('sandwich-joint/native', 'sandwich-dense5/native'), ('sandwich-sw_stack_mix/native', 'sandwich-sw_stack/native'),
          ('sandwich-sw_stack_mix/swap', 'sandwich-sw_stack_mix/native'), ('sandwich-sw_stack/swap', 'sandwich-sw_stack/native'),
          ('sandwich-joint/swap', 'sandwich-joint/native'), ('sandwich-sw_stack/native', 'sandwich-sw_warm/native')]
         + [(f'{s}-tied_{o}_{m}/native', f'{s}-slot_{o}_{m}/native') for s in ('translator_s0', 'translator_s1')
            for o in ('fixed', 'mix') for m in ('residual', 'attnres')]
         + [(f'{s}-{t}_attnres/native', f'{s}-{t}_residual/native') for s in ('translator_s0', 'translator_s1')
            for t in ('tied_fixed', 'tied_mix', 'slot_fixed', 'slot_mix')]
         + [(f'{d}-oracle/B', f'{d}-oracle/A') for d in ('domain_1x', 'domain_8x')]
         + [(f'{d}-oracle/{r}', f'{d}-single/A') for d in ('domain_1x', 'domain_8x') for r in ('A', 'B')]
         + [('domain_1x-both/both', 'domain_1x-single/A'), ('domain_1x-random/A', 'domain_1x-single/A'),
            ('domain_8x-single/A', 'domain_1x-single/A')])


def load(root):
    data, acc = {}, {}
    for d in sorted(p for p in root.iterdir() if (p/'summary.json').is_file()):
        s = json.loads((d/'summary.json').read_text())
        for route, digest in s['items_sha256'].items():
            f = d/f'items_{route}.npz'
            if hashlib.sha256(f.read_bytes()).hexdigest() != digest:
                raise ValueError(f'items hash {d.name}/{route}')
            with np.load(f) as z:
                items = {t: dict(id=z[f'{t}_id'], correct=z[f'{t}_correct']) for t in TASKS}
            key = f'{d.name}/{route}'
            for t in TASKS:
                a = float(items[t]['correct'].mean())
                if abs(a - s['tasks'][route][t]['accuracy']) > 1e-12:
                    raise ValueError(f'accuracy replay {key}/{t}')
            data[key] = items
            acc[key] = {t: float(items[t]['correct'].mean()) for t in TASKS}
    return data, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/family_downstream_20261123'))
    a = p.parse_args()
    data, acc = load(a.root)
    pairs = {}
    for x, y in PAIRS:
        if x not in data or y not in data:
            continue
        pairs[f'{x} - {y}'] = {}
        for t in TASKS:
            if not np.array_equal(data[x][t]['id'], data[y][t]['id']):
                raise ValueError(f'item ids differ {x} {y} {t}')
            pairs[f'{x} - {y}'][t] = paired_accuracy(data[x][t]['correct'], data[y][t]['correct'])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'replay.json').write_text(json.dumps(dict(accuracy=acc, pairs=pairs), indent=1))
    print(f"{'model/route':44s} " + ' '.join(f'{t[:9]:>9s}' for t in TASKS))
    for k, v in acc.items():
        print(f'{k:44s} ' + ' '.join(f'{100*v[t]:9.2f}' for t in TASKS))
    print()
    for k, v in pairs.items():
        print(f'{k:72s} ' + ' '.join(f"{100*v[t]['delta']:+6.2f}[{100*v[t]['lo']:+.1f},{100*v[t]['hi']:+.1f}]" for t in TASKS))


if __name__ == '__main__':
    main()
