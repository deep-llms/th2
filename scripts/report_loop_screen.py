"""Local replay/report for the loop screen: paired bootstrap CIs, pass curves, calibration.

Usage: python3 scripts/report_loop_screen.py <export_dir> [--out results/loop_screen_20261123]
<export_dir> holds <arm>/summary.json, <arm>/scores.npz, <arm>/calibration.jsonl
(flattened Dropbox export folders are accepted via --layout flat).
Decision rule (docs/LOOP_SCREEN_20261123.md): |delta| >= 0.02 nat is a signal.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from efficient_llm.warm_order.loop_screen import ARMS, LOOP_ARMS

THRESHOLD = 0.02


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(root, arm):
    d = Path(root)/arm
    summary = json.loads((d/'summary.json').read_text())
    if sha(d/'scores.npz') != summary['score_sha256']:
        raise ValueError('score hash mismatch: '+arm)
    with np.load(d/'scores.npz') as z:
        scores = {k: z[k] for k in z.files}
    for k, v in summary['nll'].items():
        if abs(float(scores[k].mean()) - v) > 1e-9:
            raise ValueError(f'replay mismatch {arm}/{k}')
    cal = [json.loads(l) for l in (d/'calibration.jsonl').read_text().splitlines()]
    return summary, scores, cal


def paired_ci(a, b, reps=10000, seed=0):
    diff = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (reps, len(diff)))].mean(1)
    return float(diff.mean()), float(np.quantile(boots, .025)), float(np.quantile(boots, .975)), int((diff < 0).sum())


def verdict(delta):
    if delta <= -THRESHOLD:
        return 'better (signal)'
    if delta >= THRESHOLD:
        return 'worse (signal)'
    return 'inconclusive'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/loop_screen_20261123'))
    a = p.parse_args()
    data = {arm: load(a.root, arm) for arm in ARMS if (a.root/arm).is_dir()}
    if 'dense4' not in data:
        raise SystemExit('dense4 baseline missing')
    base = data['dense4'][1]['native']
    rows, curves = [], {}
    for arm, (summary, scores, cal) in data.items():
        if not np.array_equal(scores['row_ids'], data['dense4'][1]['row_ids']):
            raise ValueError('row ids differ: '+arm)
        mean, lo, hi, wins = paired_ci(scores['native'], base)
        row = dict(arm=arm, nll=float(scores['native'].mean()), delta_vs_dense4=mean, ci95=[lo, hi],
                   rows_better=wins, verdict=verdict(mean) if arm != 'dense4' else 'baseline',
                   block_apps=summary['block_applications_per_token'], params=summary['parameters'],
                   non_embedding_params=summary['non_embedding_parameters'], steps=summary['step'],
                   tokens_per_s=summary['tokens_per_s'], train_seconds=summary['train_seconds'])
        if 'swap' in scores:
            row['swap_cost'] = float(scores['swap'].mean() - scores['native'].mean())
        if arm in LOOP_ARMS:
            row['pass_curve'] = {k: float(scores[f'native_p{k}'].mean()) for k in range(1, 7)}
        if 'memory_off' in scores:
            row['memory_off_cost'] = float(scores['memory_off'].mean() - scores['native'].mean())
        for key in ('message_gate', 'attnres_query_norm'):
            if key in summary:
                row[key] = summary[key]
        rows.append(row)
        curves[arm] = cal
    pairs = {}
    names = [r['arm'] for r in rows]
    for x, y in [('gated', 'serial'), ('gated', 'memory'), ('memory', 'serial'), ('gated_var', 'gated'),
                 ('dense8', 'dense4_2x'), ('gated', 'dense8'), ('gated', 'dense4_2x'), ('attnres', 'dense4')]:
        if x in names and y in names:
            m, lo, hi, w = paired_ci(data[x][1]['native'], data[y][1]['native'])
            pairs[f'{x}-{y}'] = dict(delta=m, ci95=[lo, hi], rows_better=w, verdict=verdict(m))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'replay.json').write_text(json.dumps(dict(threshold=THRESHOLD, arms=rows, pairs=pairs,
                                                     calibration=curves), indent=1))
    print(f"{'arm':10s} {'NLL':>9s} {'Δ dense4':>9s} {'95% CI':>19s} {'verdict':>16s} apps  swap")
    for r in rows:
        ci = f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]"
        print(f"{r['arm']:10s} {r['nll']:9.5f} {r['delta_vs_dense4']:+9.5f} {ci:>19s} {r['verdict']:>16s} "
              f"{r['block_apps']:4d} {r.get('swap_cost', float('nan')):+.4f}")
    for k, v in pairs.items():
        print(f"{k:20s} {v['delta']:+.5f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] {v['verdict']}")


if __name__ == '__main__':
    main()
