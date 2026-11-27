"""Local replay for the DOT sandwich round (docs/DOT_SANDWICH_20261123.md): NLL and downstream vs the standard transformer.

Usage: PYTHONPATH=. python3 scripts/report_dot_sandwich.py <verified_dir> [--out results/dot_sandwich_20261123]
<verified_dir>: s{seed}/{arm}/{summary.json,scores.npz,calibration.jsonl} and downstream/s{seed}-{arm}/{summary.json,items.npz}.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from efficient_llm.warm_order.loop_screen_downstream import TASKS, paired_accuracy
from scripts.report_scale_round import boot, load

R1, R2, R4 = (Path(f'results/{n}_20261123/verified') for n in ('loop_screen', 'loop_screen2', 'loop_screen4'))
DENSE = {0: dict(dense4=R1/'s0/dense4', dense8=R1/'s0/dense8'), 1: dict(dense4=R4/'s1/dense4', dense8=R2/'s1/dense8')}
D1, D2, D4 = (Path(f'results/{n}_20261123/verified') for n in ('loop_screen_downstream', 'loop_screen2_downstream', 'loop_screen4_downstream'))
DENSE_DS = {0: dict(dense4=D1/'dense4', dense8=D1/'dense8'), 1: dict(dense4=D4/'s1-dense4', dense8=D2/'s1-dense8')}


def items(d):
    d = Path(d)
    s = json.loads((d/'summary.json').read_text())
    if hashlib.sha256((d/'items.npz').read_bytes()).hexdigest() != s['items_sha256']:
        raise ValueError(f'items hash {d}')
    with np.load(d/'items.npz') as z:
        return {t: (z[f'{t}_id'], z[f'{t}_correct']) for t in TASKS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/dot_sandwich_20261123'))
    a = p.parse_args()
    out = {}
    fixed = {}
    for d in sorted(a.root.glob('s*/*')):
        if not (d/'summary.json').is_file():
            continue
        seed, arm = int(d.parent.name[1:]), d.name
        summary, s, _ = load(d)
        row = dict(nll=float(s['native'].mean()), routes=summary.get('dot_routes'),
                   tokens_per_s=summary.get('tokens_per_s'), vs={}, downstream={})
        for name, ref in DENSE[seed].items():
            r = load(ref)[1]
            if not np.array_equal(r['row_ids'], s['row_ids']):
                raise ValueError('row ids differ')
            row['vs'][name] = dict(boot(s['native'] - r['native']), ref_nll=float(r['native'].mean()))
        if arm == 'sw_fixed':
            fixed[seed] = s['native']
        ds = a.root/'downstream'/f's{seed}-{arm}'
        if (ds/'items.npz').is_file():
            x = items(ds)
            row['accuracy'] = {t: float(x[t][1].mean()) for t in TASKS}
            for name, ref in DENSE_DS[seed].items():
                y = items(ref)
                row['downstream'][name] = {t: paired_accuracy(x[t][1], y[t][1]) for t in TASKS
                                           if np.array_equal(x[t][0], y[t][0])}
        out[f's{seed}/{arm}'] = row
    for k, row in out.items():
        seed = int(k[1])
        if seed in fixed and not k.endswith('sw_fixed'):
            s = load(a.root/k)[1]['native']
            row['vs']['sw_fixed'] = boot(s - fixed[seed])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'replay.json').write_text(json.dumps(out, indent=1))
    for k, r in out.items():
        print(f"{k:20s} {r['nll']:.4f}  " + '  '.join(
            f"vs {n} {c['delta']:+.4f}[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]" for n, c in r['vs'].items()))
        if r.get('routes'):
            print(f"    routes: distinct {r['routes']['distinct_orders']}, modal share {r['routes']['modal_share']:.3f}, "
                  f"usage {[[round(u, 2) for u in step[:3]] for step in r['routes']['usage']]}")
        for n, t in r['downstream'].items():
            print(f"    downstream vs {n}: " + '  '.join(f"{task[:6]} {100*v['delta']:+.2f}[{100*v['lo']:+.1f},{100*v['hi']:+.1f}]" for task, v in t.items()))


if __name__ == '__main__':
    main()
