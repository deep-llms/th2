"""Local synthesis of authenticated small exports; never loads model weights.

Run after the two archive verifiers. Quantiles describe these recorded rows,
not population confidence intervals or percentages of causal responsibility.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(root):
    output = dict(schema=1, seeds={}, source_hashes={},
                  scope='Exploratory DCLM, two parent seeds; historical overlap unknown; no causal shares')
    for seed in (0, 1):
        study = root/'adaptation_export'/'study'/f's{seed}'
        report = read(study/'REPORT.json')
        traces = [[json.loads(line) for line in
                   (root/'adaptation_export'/'runs'/f'{b}-s{seed}'/'trace.jsonl').read_text().splitlines()]
                  for b in ('A', 'B')]
        assert len(traces[0]) == len(traces[1]) == 2000
        for step, (a, b) in enumerate(zip(*traces), 1):
            assert a['step'] == b['step'] == step
            assert all(a[k] == b[k] for k in ('rows', 'parent_step', 'lr'))
            assert a['path'] == [0, 1, 2, 3] and b['path'] == [0, 2, 1, 3]
        panel = study/'step_002000'/'loss'
        labels = read(panel/'manifest.json')['cases']
        with np.load(panel/'scores.npz', allow_pickle=False) as raw:
            counts = raw['0/token_count'].copy()
            sums = {label: raw[f'{i}/nll_sum'].copy() for i, label in enumerate(labels)}
        # Independent fsum means check against the verified terminal report.
        for label, values in sums.items():
            assert abs(math.fsum(values)/int(counts.sum()) - report['points'][-1]['nll'][label]) < 1e-11

        def contrast(terms):
            v = sum(coef*sums[label] for label, coef in terms.items())
            per_row = v/counts
            return dict(mean_nat=math.fsum(v)/int(counts.sum()),
                        q10_median_q90=np.quantile(per_row, [.1, .5, .9]).tolist(),
                        positive_rows=int((per_row > 0).sum()), negative_rows=int((per_row < 0).sum()),
                        rows=len(counts))

        effects = {
            'remaining_gap': contrast({'B/swap': 1, 'A/canonical': -1}),
            'canonical_cost_of_B_adaptation': contrast({'B/canonical': 1, 'parent/canonical': -1})}
        for branch in ('A', 'B'):
            p = 'weights/'+branch+'/'
            for name, terms in {
                'pair_only': {'00': 1, '10': -1},
                'rest_only': {'00': 1, '01': -1},
                'joint': {'00': 1, '11': -1},
                'pair_given_rest': {'01': 1, '11': -1},
                'rest_given_pair': {'10': 1, '11': -1},
                'interaction': {'01': 1, '11': -1, '00': -1, '10': 1},
            }.items():
                effects[branch+'/'+name] = contrast({p+k: v for k, v in terms.items()})
        geometry = []
        for key in report['points'][-1]['geometry']['immediate']['metrics']:
            geometry.append(dict(key=key, comparisons={
                name: {k: item['metrics'][key][k] for k in
                       ('centered_linear_cka', 'mean_cosine', 'mean_squared_distance', 'mean_shift_squared')}
                for name, item in report['points'][-1]['geometry'].items()}))
        background_dir = root/'path_background_export'/'prepared'/f's{seed}'/'study'
        background = read(background_dir/'REPORT.json')
        with np.load(background_dir/'scores.npz', allow_pickle=False) as raw:
            for label, item in background['scores']['cases'].items():
                assert abs(math.fsum(raw[label])/int(raw['token_count'].sum())-item['nll']) < 1e-11
        output['seeds'][str(seed)] = dict(
            paired_training_steps_checked=2000,
            trajectory=[{k: p[k] for k in ('step', 'ordinary_gain_nat', 'swapped_gain_nat',
                                           'excess_gain_over_ordinary_nat', 'remaining_gap_nat')}
                        for p in report['points']],
            terminal_nll=report['points'][-1]['nll'], terminal_effects=effects,
            parent_component_effects={k: v for k, v in report['points'][-1]['contrasts_nat_per_token'].items()
                                      if k.startswith('component/')},
            geometry=geometry, background=background)
        for p in (study/'REPORT.json', panel/'scores.npz', panel/'manifest.json',
                  background_dir/'REPORT.json', background_dir/'scores.npz'):
            output['source_hashes'][str(p.relative_to(root))] = digest(p)
        for b in ('A', 'B'):
            p = root/'adaptation_export'/'runs'/f'{b}-s{seed}'/'trace.jsonl'
            output['source_hashes'][str(p.relative_to(root))] = digest(p)
    return output


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    colors = ['#2166ac', '#b35806']
    for seed, color in zip(('0', '1'), colors):
        item = summary['seeds'][seed]
        points = item['trajectory']
        axes[0].plot([p['step'] for p in points], [p['remaining_gap_nat'] for p in points],
                     'o-', label='Seed '+seed, color=color)
        width = .36; offset = (int(seed)-.5)*width
        for b, marker in [('B', 'o'), ('A', 'x')]:
            vals = [item['terminal_effects'][b+'/'+k]['mean_nat'] for k in ('pair_only', 'rest_only', 'joint')]
            axes[1].plot(np.arange(3)+offset, vals, marker+('--' if b == 'A' else '-'), color=color,
                         label=f'{b} weights, seed {seed}')
        cases = item['background']['scores']['cases']
        values = [-cases['B/'+k]['gain_vs_same_path_identity'] for k in ('project', 'project_norm_restored')]
        rand = [-cases[f'B/random_energy_matched/{s}']['gain_vs_same_path_identity'] for s in (4101,4102,4103)]
        axes[2].plot(np.arange(2)+offset, values, 'o', color=color, label='Seed '+seed)
        axes[2].plot(np.full(3, 2+offset), rand, 'x', color=color)
    axes[0].set(xlabel='Continuation updates', ylabel='B/swap − A/canonical NLL (nat/token)', title='Matched continuation: gap persists')
    axes[0].set_xscale('symlog', linthresh=10);axes[0].legend()
    axes[1].set(xticks=range(3), xticklabels=['Pair only', 'Rest only', 'Both'],
                ylabel='NLL reduction vs parent/swap', title='Terminal weight substitution, no refitting')
    axes[1].legend(fontsize=8)
    axes[2].set(xticks=range(3), xticklabels=['Null SVD', '+ norm restore', 'Matched random'],
                ylabel='B NLL increase (harm)', title='Frozen null-input pilot, 128 rows')
    axes[2].legend()
    for ax in axes:
        ax.axhline(0, color='gray', linewidth=.7);ax.grid(alpha=.18)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/layer_dependency_20261120'))
    args = p.parse_args();result = summarize(args.root)
    out = args.root/'adaptation_and_background_summary.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    plot(result, args.root/'adaptation_and_background.png')
    print(json.dumps(dict(status='PASS', summary=str(out), paired_trace_rows_checked=4000,
                         geometry_raw_moments_replayed_locally=False, model_execution=False)))


if __name__ == '__main__':
    main()
