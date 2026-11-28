"""Paired downstream comparison of local runs (user, 27-28 Nov): the 3-task average (HellaSwag, PIQA, ARC-Easy) with
item-paired 95% CIs per seed and pooled over seeds; LAMBADA reported separately (it behaves like NLL), never averaged in.
  python scripts/local_downstream_compare.py --ref huginn_r3_lr28 --arms armA armB ... [--base <run dir>]
"""
import argparse
from pathlib import Path

import numpy as np

TASKS = ('hellaswag', 'piqa', 'arc_easy')


def load(base, seed, arm):
    f = Path(base)/f's{seed}'/arm/'downstream_items.npz'
    return np.load(f) if f.is_file() else None


def paired(base, arm, ref, seeds, tasks):
    """Mean over tasks of the per-item accuracy difference arm - ref (percentage points) and its 95% half-width; items
    are paired within a task, tasks are independent samples (pooled seeds concatenate the paired items)."""
    means, variances = [], []
    for t in tasks:
        d = np.concatenate([load(base, s, arm)[f'{t}_correct'].astype(float) - load(base, s, ref)[f'{t}_correct'].astype(float)
                            for s in seeds])
        means.append(d.mean())
        variances.append(d.var(ddof=1) / len(d))
    return 100 * np.mean(means), 100 * 1.96 * np.sqrt(sum(variances)) / len(tasks)


def accuracy(base, seed, arm, tasks):
    return 100 * np.mean([load(base, seed, arm)[f'{t}_correct'].mean() for t in tasks])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='/disk/thuat/dot_runs_local/l1_20261126')
    p.add_argument('--ref', required=True)
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    a = p.parse_args()
    print(f'reference {a.ref}: 3-task avg', ' / '.join(f'{accuracy(a.base, s, a.ref, TASKS):.2f}' for s in a.seeds),
          '| LAMBADA', ' / '.join(f'{accuracy(a.base, s, a.ref, ("lambada_openai",)):.2f}' for s in a.seeds))
    for arm in a.arms:
        seeds = [s for s in a.seeds if load(a.base, s, arm) is not None and load(a.base, s, a.ref) is not None]
        if not seeds:
            print(f'{arm}: not scored yet')
            continue
        avg = ' / '.join(f'{accuracy(a.base, s, arm, TASKS):.2f}' for s in seeds)
        per = '  '.join('s%d %+.2f±%.2f' % ((s,) + paired(a.base, arm, a.ref, [s], TASKS)) for s in seeds)
        pooled = '%+.2f±%.2f' % paired(a.base, arm, a.ref, seeds, TASKS)
        lam = ' / '.join(f'{accuracy(a.base, s, arm, ("lambada_openai",)):.2f}' for s in seeds)
        lam_d = '%+.2f±%.2f' % paired(a.base, arm, a.ref, seeds, ('lambada_openai',))
        tasks = '  '.join('%s %+.2f' % (t, paired(a.base, arm, a.ref, seeds, (t,))[0]) for t in TASKS)
        print(f'{arm}: 3-task avg {avg} | vs ref {per} | pooled {pooled} | {tasks} | LAMBADA {lam} ({lam_d})')


if __name__ == '__main__':
    main()
