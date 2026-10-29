"""Registered FWE interim/final selections over the shared paired bootstrap."""
from eval.a2_stage1_stats import (MODELS, CONTRASTS, aligned_accuracy_arrays,
                                 paired_accuracy_intervals)

PART_MODELS = {
    'part1': ('P300-parent', 'P1T-parent', 'P300-student', 'P1T-student'),
    'part2': ('S72-parent', 'X72-student'),
    'final': MODELS,
}
PART_CONTRASTS = {
    'part1': {'P300_drop': (1, 0, -1, 0),
              'P1T_drop': (0, 1, 0, -1),
              'P1T_minus_P300_drop': (-1, 1, 1, -1)},
    'part2': {'X72_drop_from_S72': (1, -1)},
    'final': CONTRASTS,
}


def paired_report(replays, part):
    if part not in PART_MODELS:
        raise ValueError('Unregistered FWE downstream part')
    models = PART_MODELS[part]
    arrays = aligned_accuracy_arrays(replays, models=models)
    result = paired_accuracy_intervals(arrays, resamples=10000, seed=0,
                                     models=models, contrasts=PART_CONTRASTS[part])
    result.update(dataset='fwe', part=part, models=list(models))
    return result
