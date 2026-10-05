"""Measure a repeated dense-SDPA trajectory; numerical gaps are observations.

Fail for unmatched data/order/LR or incomplete training, not for a large measured
gap. Keep the old 3%/0.01 trajectory thresholds visible as descriptive flags.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def curve(arm, steps):
    ranks = [read(arm / f'rank-{i}.json') for i in range(8)]
    expected = list(range(1, steps + 1))
    for i, rank in enumerate(ranks):
        assert rank['status'] == 'passed' and rank['rank'] == i and rank['world_size'] == 8
        assert rank['mode'] == arm.name and rank['tokens_per_update'] == 1048576
        assert rank['observed_rows'] == steps * 64
        assert [row['step'] for row in rank['steps']] == expected
    history = [row for row in ranks[0]['log_history'] if 'loss' in row]
    assert [row['step'] for row in history] == expected
    assert all(math.isfinite(row[key]) for row in history for key in ('loss', 'grad_norm', 'learning_rate'))
    assert read(arm / 'trainer_state.json')['global_step'] == steps
    return ranks, history


def pair(reference, candidate):
    rows = []
    for a, b in zip(reference, candidate, strict=True):
        assert a['step'] == b['step'] and a['learning_rate'] == b['learning_rate']
        rows.append(dict(step=a['step'], reference_loss=a['loss'], candidate_loss=b['loss'],
            loss_absolute_gap=abs(a['loss']-b['loss']), reference_grad_norm=a['grad_norm'],
            candidate_grad_norm=b['grad_norm'],
            gradient_norm_relative_gap=abs(a['grad_norm']-b['grad_norm'])/max(abs(a['grad_norm']), 1e-20)))
    largest_norm = max(rows, key=lambda row: row['gradient_norm_relative_gap'])
    largest_loss = max(rows, key=lambda row: row['loss_absolute_gap'])
    above = [row['step'] for row in rows if row['gradient_norm_relative_gap'] >= .03]
    return dict(max_relative_gradient_norm_gap=largest_norm['gradient_norm_relative_gap'],
        max_gradient_gap_step=largest_norm['step'], max_gradient_gap_row=largest_norm,
        mean_relative_gradient_norm_gap=statistics.mean(row['gradient_norm_relative_gap'] for row in rows),
        median_relative_gradient_norm_gap=statistics.median(row['gradient_norm_relative_gap'] for row in rows),
        max_training_loss_gap=largest_loss['loss_absolute_gap'], max_loss_gap_step=largest_loss['step'],
        mean_training_loss_gap=statistics.mean(row['loss_absolute_gap'] for row in rows),
        first_gradient_gap_at_least_3pct=above[0] if above else None, count_gradient_gap_at_least_3pct=len(above),
        within_old_trajectory_limits=not above and largest_loss['loss_absolute_gap'] < .01,
        rows=rows)


def compare_runs(reference, repeat, steps=200):
    reference, repeat = Path(reference), Path(repeat)
    for name in ('data.json', 'eval-data.json', 'correctness.json'):
        assert (reference / name).read_bytes() == (repeat / name).read_bytes(), name
    old, a = curve(reference / 'sdpa_isolated', steps)
    new, b = curve(repeat / 'sdpa_isolated', steps)
    flash, c = curve(reference / 'fa4_isolated', steps)
    for ranks in (new, flash):
        for source, candidate in zip(old, ranks, strict=True):
            for key in ('full_input_sha256', 'input_sha256', 'observed_rows'):
                assert source[key] == candidate[key], (source['rank'], key)
    assert (repeat / 'sdpa_isolated/final_model/model.safetensors').stat().st_size > 0
    for ranks in (new, flash):
        assert ranks[0]['evaluation']['data'] == old[0]['evaluation']['data']
        assert (ranks[0]['evaluation']['common_dense']['heldout_target_tokens'] ==
                old[0]['evaluation']['common_dense']['heldout_target_tokens'])
    losses = {name: ranks[0]['evaluation']['common_dense']['heldout_lm_loss']
              for name, ranks in [('original_sdpa', old), ('repeat_sdpa', new), ('original_fa4', flash)]}
    assert all(math.isfinite(loss) for loss in losses.values())
    return dict(status='completed', matched_streams_and_learning_rates=True, steps=steps,
        reference=str(reference), repeat=str(repeat), heldout_losses=losses,
        dense_repeat=pair(a, b), original_fa4_comparison=pair(a, c),
        gradient_gap_definition='abs(candidate_norm-reference_norm)/abs(reference_norm)',
        note='One repeat; a trajectory gradient-norm difference is not same-weight gradient error.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--repeat', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare_runs(args.reference, args.repeat)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('SDPA_REPEAT_COMPARISON', json.dumps({k: ({x: y for x, y in v.items() if x != 'rows'}
        if isinstance(v, dict) else v) for k, v in result.items()}), flush=True)


if __name__ == '__main__':
    main()
