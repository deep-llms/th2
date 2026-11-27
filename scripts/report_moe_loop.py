"""Local replay for the MoE-in-loop + bounded dynamic AttnRes round (docs/MOE_LOOP_ROUND_20261123.md).

Usage: PYTHONPATH=. python3 scripts/report_moe_loop.py <verified_dir> [--out results/moe_loop_20261123]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_scale_round import boot, load

R1, R2, R4 = (Path(f'results/{n}_20261123/verified') for n in ('loop_screen', 'loop_screen2', 'loop_screen4'))
REFS = {  # verified static references on the same rows: {seed: {name: path}}
    0: dict(dense4=R1/'s0/dense4', attnres=R1/'s0/attnres', dense8=R1/'s0/dense8',
            loop_attnres=R2/'s0/loop_attnres', attnres8=R2/'s0/attnres8'),
    1: dict(dense4=R4/'s1/dense4', attnres=R4/'s1/attnres', dense8=R2/'s1/dense8',
            loop_attnres=R4/'s1/loop_attnres', attnres8=R4/'s1/attnres8'),
}
CONTRASTS = {
    'attnres_moe': ('attnres', 'dense4', 'attnres8'),
    'loop_attnres_moe': ('loop_attnres', 'attnres8', 'dense8'),
    'dense4_moe': ('dense4', 'attnres'),
    'attnres8_moe': ('attnres8', 'dense8'),
    'dynattn_tanh': ('attnres', 'dense4'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/moe_loop_20261123'))
    a = p.parse_args()
    out = {}
    for d in sorted(a.root.glob('s*/*')):
        if not (d/'summary.json').is_file():
            continue
        seed, arm = int(d.parent.name[1:]), d.name
        summary, s, _ = load(d)
        row = dict(nll=float(s['native'].mean()), vs={}, moe_usage=summary.get('moe_usage'),
                   dyn_b_norm=summary.get('attnres_dyn_b_norm'),
                   swap_cost=float(s['swap'].mean() - s['native'].mean()) if 'swap' in s else None,
                   pass_curve={k: float(s[f'native_p{k}'].mean()) for k in range(1, 7)} if 'native_p1' in s else None)
        for ref in CONTRASTS[arm]:
            r = load(REFS[seed][ref])[1]
            if not np.array_equal(r['row_ids'], s['row_ids']):
                raise ValueError(f'row ids differ {arm} {ref}')
            row['vs'][ref] = dict(boot(s['native'] - r['native']), ref_nll=float(r['native'].mean()))
        out[f's{seed}/{arm}'] = row
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'replay.json').write_text(json.dumps(out, indent=1))
    for k, r in out.items():
        print(f"{k:20s} {r['nll']:.4f}  " + '  '.join(
            f"vs {n} ({c['ref_nll']:.4f}) {c['delta']:+.4f}[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]" for n, c in r['vs'].items()))
        if r['moe_usage']:
            print('    usage', {b: [round(u, 3) for u in v] for b, v in r['moe_usage'].items()})
        if r['dyn_b_norm']:
            print('    dyn B norm mean', round(float(np.mean(r['dyn_b_norm'])), 3))


if __name__ == '__main__':
    main()
