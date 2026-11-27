"""Local replay for the data-scale round (docs/SCALE_ROUND_20261123.md).

Usage: PYTHONPATH=. python3 scripts/report_scale_round.py <verified_dir> [--out results/scale_round_20261123]
<verified_dir> holds the extracted archive: loop/s0/<arm>/..., domain/<arm>/...
Every score file is hash-checked against its summary and every summary mean is replayed.
Cross-scale contrasts are paired on the same measurement rows against the verified 1x/2x seed-0 runs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

THRESHOLD = 0.02
LOOP_1X = {  # seed-0 verified exports, same 512 registered measurement rows
    'dense4': 'results/loop_screen_20261123/verified/s0/dense4',
    'attnres': 'results/loop_screen_20261123/verified/s0/attnres',
    'dense8': 'results/loop_screen_20261123/verified/s0/dense8',
    'loop_attnres': 'results/loop_screen2_20261123/verified/s0/loop_attnres',
    'attnres8': 'results/loop_screen2_20261123/verified/s0/attnres8',
}
LOOP_2X = {
    'dense4': 'results/loop_screen_20261123/verified/s0/dense4_2x',
    'attnres': 'results/loop_screen2_20261123/verified/s0/attnres_2x',
    'dense8': 'results/loop_screen4_20261123/verified/s0/dense8_2x',
    'loop_attnres': 'results/loop_screen4_20261123/verified/s0/loop_attnres_2x',
}
DOMAIN_1X = 'results/domain_screen_20261123/verified'
CONTRASTS = [('loop_attnres', 'attnres'), ('loop_attnres', 'attnres8'), ('loop_attnres', 'dense8'),
             ('attnres', 'dense4'), ('attnres8', 'dense8'), ('dense8', 'dense4')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(d):
    d = Path(d)
    summary = json.loads((d/'summary.json').read_text())
    if sha(d/'scores.npz') != summary['score_sha256']:
        raise ValueError(f'score hash mismatch: {d}')
    with np.load(d/'scores.npz') as z:
        scores = {k: z[k] for k in z.files}
    for k, v in summary['nll'].items():
        if abs(float(scores[k].mean()) - v) > 1e-9:
            raise ValueError(f'replay mismatch {d}/{k}')
    cal = [json.loads(line) for line in (d/'calibration.jsonl').read_text().splitlines()] \
        if (d/'calibration.jsonl').is_file() else []
    return summary, scores, cal


def boot(diff, reps=10000, seed=0):
    diff = np.asarray(diff, np.float64)
    rng = np.random.default_rng(seed)
    b = diff[rng.integers(0, len(diff), (reps, len(diff)))].mean(1)
    return dict(delta=float(diff.mean()), ci95=[float(np.quantile(b, .025)), float(np.quantile(b, .975))],
                rows_better=int((diff < 0).sum()))


def label(delta):
    return 'better (signal)' if delta <= -THRESHOLD else 'worse (signal)' if delta >= THRESHOLD else 'inconclusive'


def loop_part(root):
    arms = {a.removesuffix('_4x'): load(root/'loop'/'s0'/a)
            for a in ('loop_attnres_4x', 'attnres_4x', 'attnres8_4x', 'dense8_4x', 'dense4_4x')}
    ref1 = {a: load(p) for a, p in LOOP_1X.items()}
    ref2 = {a: load(p) for a, p in LOOP_2X.items()}
    ids = arms['dense4'][1]['row_ids']
    for a, (_, s, _) in {**arms, **ref1, **ref2}.items():
        if not np.array_equal(s['row_ids'], ids):
            raise ValueError('row ids differ: '+a)
    nat = {a: v[1]['native'].astype(np.float64) for a, v in arms.items()}
    nat1 = {a: v[1]['native'].astype(np.float64) for a, v in ref1.items()}
    nat2 = {a: v[1]['native'].astype(np.float64) for a, v in ref2.items()}
    out = dict(nll={a: float(v.mean()) for a, v in nat.items()},
               nll_1x={a: float(v.mean()) for a, v in nat1.items()},
               nll_2x={a: float(v.mean()) for a, v in nat2.items()}, contrasts={})
    for x, y in CONTRASTS:
        c = boot(nat[x] - nat[y])
        c['verdict'] = label(c['delta'])
        if x in nat1 and y in nat1:
            c['delta_1x'] = float((nat1[x] - nat1[y]).mean())
            c['change_4x_minus_1x'] = boot((nat[x] - nat[y]) - (nat1[x] - nat1[y]))
        if x in nat2 and y in nat2:
            c['delta_2x'] = float((nat2[x] - nat2[y]).mean())
        out['contrasts'][f'{x}-{y}'] = c
    la = arms['loop_attnres'][1]
    out['loop_pass_curve'] = {k: float(la[f'native_p{k}'].mean()) for k in range(1, 7)}
    out['swap_cost'] = {a: float(v[1]['swap'].mean() - v[1]['native'].mean()) for a, v in arms.items() if 'swap' in v[1]}
    out['calibration'] = {a: [(c['step'], c['nll']) for c in v[2]] for a, v in arms.items()}
    return out


def domain_part(root):
    single, oracle = load(root/'domain'/'single')[1], load(root/'domain'/'oracle')[1]
    s1, o1 = load(Path(DOMAIN_1X)/'single')[1], load(Path(DOMAIN_1X)/'oracle')[1]
    out = {}
    for dom, native in (('dclm', 'A'), ('fwe', 'B')):
        lab = oracle[f'{dom}_measure_{native}'].astype(np.float64)
        sin = single[f'{dom}_measure_A'].astype(np.float64)
        lab1 = o1[f'{dom}_measure_{native}'].astype(np.float64)
        sin1 = s1[f'{dom}_measure_A'].astype(np.float64)
        other = oracle[f'{dom}_measure_{"B" if native == "A" else "A"}'].astype(np.float64)
        per_row = np.minimum(lab, other)
        c = boot(lab - sin)
        c['verdict'] = label(c['delta'])
        out[dom] = dict(single=float(sin.mean()), oracle_label=float(lab.mean()),
                        oracle_other_middle=float(other.mean()), oracle_minus_single=c,
                        oracle_minus_single_1x=float((lab1 - sin1).mean()),
                        change_8x_minus_1x=boot((lab - sin) - (lab1 - sin1)),
                        per_row_oracle_minus_label=float((per_row - lab).mean()),
                        learned_minus_label=float(oracle[f'{dom}_measure_learned'].mean() - lab.mean()),
                        specialization_gap=float(other.mean() - lab.mean()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/scale_round_20261123'))
    a = p.parse_args()
    report = dict(threshold=THRESHOLD, loop_4x=loop_part(a.root), domain_8x=domain_part(a.root))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'replay.json').write_text(json.dumps(report, indent=1))
    lp, dp = report['loop_4x'], report['domain_8x']
    for k, c in lp['contrasts'].items():
        ch = c.get('change_4x_minus_1x')
        print(f"{k:24s} 4x {c['delta']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] {c['verdict']:16s}"
              f" 2x {c.get('delta_2x', float('nan')):+.4f} 1x {c.get('delta_1x', float('nan')):+.4f}"
              + (f"  change {ch['delta']:+.4f} [{ch['ci95'][0]:+.4f},{ch['ci95'][1]:+.4f}]" if ch else ''))
    print('pass curve', {k: round(v, 4) for k, v in lp['loop_pass_curve'].items()})
    for dom, r in dp.items():
        c, ch = r['oracle_minus_single'], r['change_8x_minus_1x']
        print(f"{dom}: oracle-single {c['delta']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] (1x {r['oracle_minus_single_1x']:+.4f}),"
              f" change {ch['delta']:+.4f} [{ch['ci95'][0]:+.4f},{ch['ci95'][1]:+.4f}], per-row {r['per_row_oracle_minus_label']:+.4f},"
              f" learned {r['learned_minus_label']:+.4f}, specialization gap {r['specialization_gap']:+.4f}")


if __name__ == '__main__':
    main()
