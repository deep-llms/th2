"""Local replay for MoE at 4x data (docs/MOE_SCALE_20261123.md).

Usage: PYTHONPATH=. python3 scripts/report_moe_scale.py <verified_dir> [--out results/moe_scale_20261123]
Seed-0 static 4x references come from the verified scale round; seed-1 static references are in this round.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_scale_round import boot, load

SCALE0 = Path('results/scale_round_20261123/verified/loop/s0')
ONE_X = {  # 1x MoE gains (docs/MOE_LOOP_ROUND_20261123.md), for the change column
    ('attnres_moe_4x', 0): -0.0327, ('attnres_moe_4x', 1): -0.0558,
    ('loop_attnres_moe_4x', 0): -0.0269, ('loop_attnres_moe_4x', 1): -0.0209, ('attnres8_moe_4x', 0): -0.0372,
}
PAIRS = {'attnres_moe_4x': ('attnres_4x',), 'loop_attnres_moe_4x': ('loop_attnres_4x', 'attnres8_4x'),
         'attnres8_moe_4x': ('attnres8_4x',)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/moe_scale_20261123'))
    a = p.parse_args()
    out = {}
    for d in sorted(a.root.glob('s*/*')):
        if not (d/'summary.json').is_file():
            continue
        seed, arm = int(d.parent.name[1:]), d.name
        summary, s, cal = load(d)
        row = dict(nll=float(s['native'].mean()), vs={}, calibration=[(c['step'], c['nll']) for c in cal])
        for ref in PAIRS.get(arm, ()):
            rdir = SCALE0/ref if seed == 0 else a.root/'s1'/ref
            r = load(rdir)[1]
            if not np.array_equal(r['row_ids'], s['row_ids']):
                raise ValueError(f'row ids differ {arm} {ref}')
            row['vs'][ref] = dict(boot(s['native'] - r['native']), ref_nll=float(r['native'].mean()))
        if (arm, seed) in ONE_X and PAIRS[arm][0] in row['vs']:
            row['gain_1x'] = ONE_X[(arm, seed)]
        out[f's{seed}/{arm}'] = row
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'replay.json').write_text(json.dumps(out, indent=1))
    for k, r in out.items():
        print(f"{k:24s} {r['nll']:.4f}  " + '  '.join(
            f"vs {n} ({c['ref_nll']:.4f}) {c['delta']:+.4f}[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]" for n, c in r['vs'].items())
              + (f"  (1x gain {r['gain_1x']:+.4f})" if 'gain_1x' in r else ''))


if __name__ == '__main__':
    main()
