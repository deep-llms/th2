"""Local replay for the MoE-control + dynamic AttnRes round (docs/MOE_DYN_ROUND_20261123.md).

Usage: PYTHONPATH=. python3 scripts/report_moe_dyn.py <verified_dir> [--out results/moe_dyn_20261123]
Hash-checks and replays every summary, then pairs each arm with its verified same-data reference.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_scale_round import boot, load

DOMAIN = Path('results/domain_screen_20261123/verified')
ROUTER = Path('results/router_screen_20261123/verified')
LOOP_REF = {  # (seed, dynamic arm) -> verified static AttnRes reference on the same rows
    (0, 'dynattn'): 'results/loop_screen_20261123/verified/s0/attnres',
    (0, 'dynattn8'): 'results/loop_screen2_20261123/verified/s0/attnres8',
    (0, 'loop_dynattn'): 'results/loop_screen2_20261123/verified/s0/loop_attnres',
    (1, 'dynattn'): 'results/loop_screen4_20261123/verified/s1/attnres',
    (1, 'loop_dynattn'): 'results/loop_screen4_20261123/verified/s1/loop_attnres',
}
DENSE_REF = {0: ('results/loop_screen_20261123/verified/s0/dense4', 'results/loop_screen_20261123/verified/s0/dense8'),
             1: ('results/loop_screen4_20261123/verified/s1/dense4', 'results/loop_screen2_20261123/verified/s1/dense8')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/moe_dyn_20261123'))
    a = p.parse_args()
    single = load(DOMAIN/'single')[1]
    tok = load(ROUTER/'tok_moe2')[1]
    out = dict(moe={}, dyn={})
    for arm in ('moe2_top1', 'moe4_top1', 'moe8_top2'):
        summary, s, _ = load(a.root/'moe'/arm)
        row = dict(usage=summary['usage'], choice=summary['choice'])
        for d in ('dclm', 'fwe'):
            r = s[f'{d}_measure_routed'].astype(np.float64)
            row[d] = dict(nll=float(r.mean()), vs_single=boot(r - single[f'{d}_measure_A']),
                          vs_tok_moe2_sampled=boot(r - tok[f'{d}_measure_routed']))
        out['moe'][arm] = row
    for (seed, arm), ref in LOOP_REF.items():
        summary, s, cal = load(a.root/'loop'/f's{seed}'/arm)
        r_s = load(Path(ref))[1]
        if not np.array_equal(s['row_ids'], r_s['row_ids']):
            raise ValueError('row ids differ: '+arm)
        dense = load(Path(DENSE_REF[seed][1 if arm == 'dynattn8' or arm.startswith('loop') else 0]))[1]
        out['dyn'][f's{seed}/{arm}'] = dict(
            nll=float(s['native'].mean()), ref_nll=float(r_s['native'].mean()),
            vs_static_attnres=boot(s['native'] - r_s['native']), vs_dense=boot(s['native'] - dense['native']),
            dyn_b_norm=summary.get('attnres_dyn_b_norm'),
            swap_cost=float(s['swap'].mean() - s['native'].mean()) if 'swap' in s else None,
            pass_curve={k: float(s[f'native_p{k}'].mean()) for k in range(1, 7)} if 'native_p1' in s else None)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'replay.json').write_text(json.dumps(out, indent=1))
    for arm, r in out['moe'].items():
        print(f"{arm:10s} " + '  '.join(f"{d} {r[d]['nll']:.4f} vs single {r[d]['vs_single']['delta']:+.4f}[{r[d]['vs_single']['ci95'][0]:+.4f},{r[d]['vs_single']['ci95'][1]:+.4f}]"
                                         f" vs tok_moe2 {r[d]['vs_tok_moe2_sampled']['delta']:+.4f}" for d in ('dclm', 'fwe')), 'usage', [round(u, 3) for u in r['usage']])
    for k, r in out['dyn'].items():
        c, dn = r['vs_static_attnres'], r['vs_dense']
        print(f"{k:16s} {r['nll']:.4f} (static {r['ref_nll']:.4f}) vs static {c['delta']:+.4f}[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]"
              f" vs dense {dn['delta']:+.4f}  B-norm mean {np.mean(r['dyn_b_norm']) if r['dyn_b_norm'] else float('nan'):.3f}")


if __name__ == '__main__':
    main()
