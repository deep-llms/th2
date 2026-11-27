"""Local replay of loop-screen downstream items: accuracies and paired item bootstrap vs a baseline arm."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from efficient_llm.warm_order.loop_screen_downstream import TASKS, paired_accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--baseline', default='dense4')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    arms = sorted(d.name for d in a.root.iterdir() if (d/'items.npz').is_file())
    data, rows = {}, []
    for arm in arms:
        s = json.loads((a.root/arm/'summary.json').read_text())
        if hashlib.sha256((a.root/arm/'items.npz').read_bytes()).hexdigest() != s['items_sha256']:
            raise ValueError('items hash '+arm)
        with np.load(a.root/arm/'items.npz') as z:
            data[arm] = {t: dict(id=z[f'{t}_id'], correct=z[f'{t}_correct']) for t in TASKS}
        for t in TASKS:
            acc = float(data[arm][t]['correct'].mean())
            if abs(acc - s['tasks'][t]['accuracy']) > 1e-12:
                raise ValueError(f'accuracy replay {arm}/{t}')
    report = {}
    print(f"{'arm':10s} " + ' '.join(f'{t[:9]:>22s}' for t in TASKS))
    for arm in arms:
        cells, report[arm] = [], {}
        for t in TASKS:
            x, b = data[arm][t], data[a.baseline][t]
            if not np.array_equal(x['id'], b['id']):
                raise ValueError('unpaired items')
            r = paired_accuracy(x['correct'], b['correct'])
            report[arm][t] = dict(accuracy=float(x['correct'].mean()), **r)
            cells.append(f"{100*x['correct'].mean():6.2f} ({100*r['delta']:+5.2f}±{50*(r['hi']-r['lo']):4.2f})")
        print(f'{arm:10s} ' + ' '.join(f'{c:>22s}' for c in cells))
    if a.out:
        a.out.write_text(json.dumps(dict(baseline=a.baseline, report=report), indent=1))


if __name__ == '__main__':
    main()
