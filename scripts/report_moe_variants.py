"""Local replay for the MoE variants round (docs/MOE_VARIANTS_20261123.md).

Usage: PYTHONPATH=. python3 scripts/report_moe_variants.py <verified_dir> [--out results/moe_variants_20261123]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_scale_round import boot, load

PRIOR = Path('results/moe_loop_20261123/verified')  # first MoE-loop run (attnres_moe, loop_attnres_moe refs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/moe_variants_20261123'))
    a = p.parse_args()
    out = {}
    for seed in (0, 1):
        new = {arm: load(a.root/f's{seed}'/arm) for arm in ('loop_attnres_moe_pr', 'loop_attnres_moe', 'attnres_moe16', 'attnres_moe_shared')}
        old = {arm: load(PRIOR/f's{seed}'/arm) for arm in ('loop_attnres_moe', 'attnres_moe')}
        rerun, first = new['loop_attnres_moe'][1]['native'], old['loop_attnres_moe'][1]['native']
        row = dict(reproduction_max_abs=float(np.abs(rerun - first).max()),
                   nll={k: float(v[1]['native'].mean()) for k, v in new.items()},
                   per_pass_router=boot(new['loop_attnres_moe_pr'][1]['native'] - rerun),
                   finer_experts=boot(new['attnres_moe16'][1]['native'] - old['attnres_moe'][1]['native']),
                   shared_expert=boot(new['attnres_moe_shared'][1]['native'] - old['attnres_moe'][1]['native']),
                   trace={k: v[0].get('moe_trace') for k, v in new.items()},
                   pass_curve={k: {p_: float(v[1][f'native_p{p_}'].mean()) for p_ in range(1, 7)}
                               for k, v in new.items() if 'native_p1' in v[1]})
        out[f's{seed}'] = row
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'replay.json').write_text(json.dumps(out, indent=1))
    for s, r in out.items():
        print(s, 'reproduction max |diff|', r['reproduction_max_abs'], {k: round(v, 4) for k, v in r['nll'].items()})
        for name in ('per_pass_router', 'finer_experts', 'shared_expert'):
            c = r[name]
            print(f"   {name:16s} {c['delta']:+.4f}[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]")
        for arm in ('loop_attnres_moe', 'loop_attnres_moe_pr'):
            for b, t in (r['trace'][arm] or {}).items():
                print(f"   {arm} block {b} consecutive overlap {[round(x, 3) for x in t['consecutive_overlap']]} (chance {t['chance']})")


if __name__ == '__main__':
    main()
