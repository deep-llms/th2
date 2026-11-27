"""Local replay for the router screen (docs/ROUTER_SCREEN_20261123.md).

Usage: PYTHONPATH=. python3 scripts/report_router_screen.py <verified_dir> [--out results/router_screen_20261123]
Hash-checks every score file, replays every summary mean, and pairs each arm with the verified 1x domain
references on the same 512 measurement rows per domain (paired row bootstrap).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_scale_round import boot, load

REF = Path('results/domain_screen_20261123/verified')
DOMAINS = ('dclm', 'fwe')
ARMS = ('seq_switch2', 'seq_switch3', 'seq_cf2', 'order_cf', 'both_mix', 'tok_moe2', 'tok_moe3')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/router_screen_20261123'))
    a = p.parse_args()
    ref = {n: load(REF/n)[1] for n in ('single', 'oracle', 'random', 'both')}
    new_single = load(a.root/'single')[1]
    out = dict(single_reproduction={k: float(np.abs(new_single[k] - ref['single'][k]).max()) for k in ref['single']}, arms={})
    for arm in ARMS:
        if not (a.root/arm/'summary.json').is_file():
            continue
        summary, s, cal = load(a.root/arm)
        forced = sorted({k.split('_measure_')[1] for k in s if '_measure_' in k} - {'routed', 'choice'})
        row = dict(nll={}, vs={}, choice=summary['choice'], usage=summary['usage'], forced={})
        for d in DOMAINS:
            routed = s[f'{d}_measure_routed'].astype(np.float64)
            row['nll'][d] = float(routed.mean())
            row['forced'][d] = {f: float(s[f'{d}_measure_{f}'].mean()) for f in forced}
            label = 'A' if d == 'dclm' else 'B'
            refs = dict(single=ref['single'][f'{d}_measure_A'], oracle_label=ref['oracle'][f'{d}_measure_{label}'],
                        random_A=ref['random'][f'{d}_measure_A'], both_fixed=ref['both'][f'{d}_measure_both'])
            row['vs'][d] = {k: boot(routed - v.astype(np.float64)) for k, v in refs.items()}
            if forced and not arm.startswith('tok'):
                stack = np.stack([s[f'{d}_measure_{f}'] for f in forced]).astype(np.float64)
                row['vs'][d]['per_row_oracle_minus_routed'] = float((stack.min(0) - routed).mean())
                row['vs'][d]['best_forced_minus_routed'] = float(stack.mean(1).min() - routed.mean())
        row['calibration_tail'] = cal[-1] if cal else None
        out['arms'][arm] = row
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'replay.json').write_text(json.dumps(out, indent=1))
    print('single reproduction max |diff|:', max(out['single_reproduction'].values()))
    for arm, r in out['arms'].items():
        for d in DOMAINS:
            v = r['vs'][d]
            print(f"{arm:12s} {d:4s} routed {r['nll'][d]:.4f}  forced {', '.join(f'{k}:{x:.4f}' for k, x in r['forced'][d].items())}")
            print('    ' + '  '.join(f"{k} {c['delta']:+.4f}[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]" for k, c in v.items() if isinstance(c, dict))
                  + ('' if 'per_row_oracle_minus_routed' not in v else
                     f"  per-row-oracle {v['per_row_oracle_minus_routed']:+.4f} best-forced {v['best_forced_minus_routed']:+.4f}"))
        print('    choice', r['choice'], 'usage', [round(u, 3) for u in r['usage']])


if __name__ == '__main__':
    main()
