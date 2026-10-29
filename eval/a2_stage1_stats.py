"""Stage1 paired item bootstrap; no model loading or GPU work.

Inputs are validated per-item primary accuracy arrays, aligned by document and
prompt by the upstream collector. Resample within each task, retaining model
pairing. The MC statistic is an equal-weight mean of the eight task accuracies.
Intervals describe benchmark-item sampling, not training-seed uncertainty.
"""
import numpy as np

MODELS = ('P300-parent', 'P1T-parent', 'S72-parent',
          'P300-student', 'P1T-student', 'X72-student')
CONTRASTS = {
    'P300_drop': (1, 0, 0, -1, 0, 0),
    'P1T_drop': (0, 1, 0, 0, -1, 0),
    'X72_drop_from_S72': (0, 0, 1, 0, 0, -1),
    'P1T_minus_P300_drop': (-1, 1, 0, 1, -1, 0),
    'X72_minus_P300_accuracy': (0, 0, 0, -1, 0, 1),
    'X72_minus_P300_drop': (-1, 0, 1, 1, 0, -1),
}


def paired_accuracy_intervals(tasks, *, resamples=10000, seed=0):
    """Each task maps all six fixed model labels to aligned binary arrays.

    Caller validates task names/counts and exact pairing before this numerical
    helper. Require eight MC tasks plus lambada_openai. Positive drop means
    lower student accuracy; the direct X72-minus-P300 contrast has the opposite
    orientation from an NLL difference, stated explicitly in output metadata.
    """
    if len(tasks) != 9 or 'lambada_openai' not in tasks:
        raise ValueError('Eight MC tasks and LAMBADA required')
    if type(resamples) is not int or resamples < 2 or type(seed) is not int:
        raise ValueError('Explicit valid bootstrap settings required')
    rng = np.random.default_rng(seed)
    weights = np.asarray(list(CONTRASTS.values()), dtype=np.float64).T
    mc_point = np.zeros(len(CONTRASTS))
    mc_draws = np.zeros((resamples, len(CONTRASTS)))
    by_task = {}

    def report(point, draws):
        lo, hi = np.quantile(draws, [.025, .975], axis=0, method='linear')
        return {name: dict(estimate=float(point[i]), ci95=[float(lo[i]), float(hi[i])])
                for i, name in enumerate(CONTRASTS)}

    for task in sorted(tasks):
        records = tasks[task]
        if set(records) != set(MODELS):
            raise ValueError('Missing or unexpected checkpoint')
        arrays = [np.asarray(records[name]) for name in MODELS]
        if any(a.ndim != 1 or a.size == 0 or a.size != arrays[0].size
               or not np.isin(a, [0, 1]).all() for a in arrays):
            raise ValueError('Expected paired finite binary item scores')
        matrix = np.stack(arrays, axis=1).astype(np.float64)
        n = len(matrix)
        # Bootstrap category counts is exactly equivalent in distribution to
        # resampling n paired item rows, with much less memory than n*10000.
        patterns, counts = np.unique(matrix, axis=0, return_counts=True)
        draws = rng.multinomial(n, counts/n, size=resamples) @ (patterns @ weights) / n
        point = matrix.mean(axis=0) @ weights
        by_task[task] = dict(examples=n, contrasts=report(point, draws),
                            accuracy=dict(zip(MODELS, map(float, matrix.mean(axis=0)))))
        if task == 'lambada_openai':
            lambada = report(point, draws)
        else:
            mc_point += point / 8
            mc_draws += draws / 8
    return dict(schema=1, resamples=resamples, seed=seed,
                method='paired item bootstrap within each task; multinomial counts of identical paired rows',
                units='accuracy fraction; multiply by100 for percentage points',
                uncertainty='benchmark items only; does not cover training seeds',
                drop_sign='parent minus student; positive is worse after healing',
                direct_sign='X72 student minus P300 student; positive is higher X72 accuracy',
                mc_mean=report(mc_point, mc_draws), lambada_accuracy=lambada, tasks=by_task)
