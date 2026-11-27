"""Learning-trajectory comparison of local runs (28 Nov 2026, user question: does DOT fall behind because the reference
learns faster later, or because DOT itself slows down?).

Every run scores the same 256 pinned calibration rows every 512 steps (calibration.jsonl). Loss curves are not lines, so
three views are used, all restricted to steps where both runs are in their constant-LR phase (DOT / dense: warm-up 128,
decay over the last 20%; Huginn: warm-up and cool-down 6.8% each):
  gap(t)        NLL_a(t) - NLL_b(t) at the same step
  multiplier(t) t_b / t, where t_b is the step at which run b reaches run a's NLL at step t (log-step interpolation of
                b's running minimum). > 1: a needs fewer steps than b for the same NLL. Falling over training means a
                loses efficiency relative to b; rising means it catches up. Independent of the curves' shapes.
  drop/doubling NLL decrease over each doubling of steps (learning speed on a log-step axis) for each run.
  python scripts/local_trajectory_analysis.py --out docs/results/local_rows_20261126/trajectories.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

BASE = Path('/disk/thuat/dot_runs_local/l1_20261126')
DOUBLINGS = (1024, 2048, 4096, 8192, 16384, 32768)


def curve(run):
    rows = [json.loads(l) for l in (BASE/run/'calibration.jsonl').read_text().splitlines()]
    s = np.array([r['step'] for r in rows if r['step'] > 0], float)
    v = np.array([r['nll'] for r in rows if r['step'] > 0], float)
    return s, v


def constant_window(run, total):
    huginn = 'huginn' in run
    lo = math.ceil(.068 * total) if huginn else 128
    hi = math.floor((1 - .068) * total) if huginn else math.floor(.8 * total)
    return lo, hi


def at(s, v, t):
    return float(np.interp(math.log(t), np.log(s), v))


def multiplier(sa, va, sb, vb, t):
    """Step at which b reaches a's NLL at step t, divided by t (b's running minimum, log-step interpolation)."""
    target = at(sa, va, t)
    mb = np.minimum.accumulate(vb)
    if target > mb[0] or target < mb[-1]:
        return None  # outside b's observed range
    i = int(np.argmax(mb <= target))
    if i == 0:
        return float(sb[0] / t)
    x0, x1, y0, y1 = math.log(sb[i - 1]), math.log(sb[i]), mb[i - 1], mb[i]
    tb = math.exp(x0 + (x1 - x0) * (y0 - target) / (y0 - y1)) if y0 != y1 else sb[i]
    return float(tb / t)


def compare(a, b, total, points):
    sa, va = curve(a)
    sb, vb = curve(b)
    lo = max(constant_window(a, total)[0], constant_window(b, total)[0])
    hi = min(constant_window(a, total)[1], constant_window(b, total)[1])
    out = []
    for t in points:
        if lo <= t <= hi:
            out.append(dict(step=t, gap=round(at(sa, va, t) - at(sb, vb, t), 4),
                            multiplier=(lambda m: None if m is None else round(m, 3))(multiplier(sa, va, sb, vb, t))))
    return dict(a=a, b=b, window=[lo, hi], final_gap=round(float(va[-1] - vb[-1]), 4), points=out)


def drops(run, total):
    s, v = curve(run)
    lo, hi = constant_window(run, total)
    return {f'{t0}-{2 * t0}': round(at(s, v, t0) - at(s, v, 2 * t0), 4)
            for t0 in DOUBLINGS if t0 >= lo and 2 * t0 <= hi}


PAIRS = [  # (a, b, total steps): same token budget, so same schedule length
    ('s0/swst_d7_tmlp_ff512_unshared_tn_lr14_fa_4x', 's0/dense8_sq_lr10_fa_4x', 49152),
    ('s1/swst_d7_tmlp_ff512_unshared_tn_lr14_fa_4x', 's0/dense8_sq_lr10_fa_4x', 49152),
    ('s0/swst_d8_tmlp_ff128_unshared_tn_lr14_fa_4x', 's0/dense8_sq_lr10_fa_4x', 49152),
    *[(f's{s}/{a}', f's{s}/{b}', 24576) for s in (0, 1) for a, b in (
        ('swst_d7_tmlp_ff512_unshared_tn_lr14_fa_2x', 'dense8_sq_lr10_fa_2x'),
        ('swst_d8_tmlp_ff128_unshared_tn_lr14_fa_2x', 'dense8_sq_lr10_fa_2x'),
        ('huginn_r3_lr28_2x', 'dense8_sq_lr10_fa_2x'),
        ('swst_d8_tmlp_ff128_unshared_tn_lr14_fa_2x', 'huginn_r3_lr28_2x'),
        ('swst_d7_tmlp_ff512_unshared_tn_lr14_fa_2x', 'huginn_r3_lr28_2x'))],
    *[(f's{s}/{a}', f's{s}/huginn_r3_lr28', 12288) for s in (0, 1) for a in (
        'dotfull_rand_inj_t512_lr14', 'dotpair_rand_inj_t512_lr14')],
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    points = [1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 39321]
    result = dict(pairs=[], drops={})
    for x, y, total in PAIRS:
        if not ((BASE/x/'summary.json').is_file() and (BASE/y/'summary.json').is_file()):
            continue
        result['pairs'].append(compare(x, y, total, points))
        for run in (x, y):
            result['drops'].setdefault(run, drops(run, total))
    if a.out:
        a.out.write_text(json.dumps(result, indent=1) + '\n')
    for c in result['pairs']:
        print(f"{c['a']}  vs  {c['b']}   window {c['window']}   final gap {c['final_gap']:+.4f}")
        print('   ' + '  '.join(f"{q['step']}: {q['gap']:+.3f} (x{q['multiplier']})" for q in c['points']))
    print('NLL drop per doubling of steps (constant-LR phase)')
    for run, d in result['drops'].items():
        print(f'   {run:52s} ' + '  '.join(f'{k}: {v:.3f}' for k, v in d.items()))


if __name__ == '__main__':
    main()
