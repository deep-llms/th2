"""CPU paired statistics for the preregistered A2 quantization experiment."""
import numpy as np

PARENTS = ('S72', 'P300', 'S188', 'P1T')
CONTRASTS = {'L_ann': (0, -1, 0, 1), 'L_stab': (-1, 0, 1, 0),
             'R_short': (-1, 1, 0, 0), 'R_long': (0, 0, -1, 1),
             'I': (1, -1, -1, 1)}
REFERENCES = {'L_ann': 'P300', 'L_stab': 'S72', 'R_short': 'S72', 'R_long': 'S188'}


def label(estimate, interval, damage):
    big, small = max(.003, .10 * damage), max(.003, .03 * damage)
    excludes = interval[0] > 0 or interval[1] < 0
    clauses = dict(at_least_T=estimate >= big, at_most_minus_T=estimate <= -big,
                   ci_excludes_zero=excludes, abs_below_t=abs(estimate) < small)
    result = ('more damage' if clauses['at_least_T'] and excludes else
              'less damage' if clauses['at_most_minus_T'] and excludes else
              'no effect' if clauses['abs_below_t'] else 'grey')
    return dict(label=result, T=big, t=small, clauses=clauses)


def nll_report(values, resamples=10000, seed=0):
    """Values: dataset -> compression (including parent) -> parent -> seq NLL."""
    datasets = {}
    for ds, compressions in values.items():
        if 'parent' not in compressions or set(compressions['parent']) != set(PARENTS): continue
        base = np.array([compressions['parent'][p] for p in PARENTS], dtype=np.float64).T
        n = len(base)
        if base.ndim != 2 or not np.isfinite(base).all(): raise ValueError('Invalid parent NLL')
        # One matrix of counts reused for every compression and model of this dataset.
        counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=resamples)
        table = {}
        for compression, parents in compressions.items():
            if compression == 'parent' or set(parents) != set(PARENTS): continue
            arr = np.array([parents[p] for p in PARENTS], dtype=np.float64).T
            if arr.shape != base.shape or not np.isfinite(arr).all(): raise ValueError('Unpaired sequence NLL')
            damage = arr - base
            means = damage.mean(0)
            draws = counts @ damage / n
            items = {}
            for name, weights in CONTRASTS.items():
                point = float(means @ weights)
                ci = np.quantile(draws @ weights, [.025, .975]).tolist()
                ref = REFERENCES.get(name)
                refdamage = float(means[PARENTS.index(ref)]) if ref else None
                entry = dict(estimate=point, ci95=ci, reference=ref, reference_damage=refdamage,
                    relative_to_reference_damage=point/refdamage if refdamage else None)
                if name in ('L_ann', 'L_stab', 'R_short'): entry.update(label(point, ci, refdamage))
                else: entry['label'] = 'descriptive only'
                items[name] = entry
            table[compression] = dict(contrasts=items, damage={p:dict(
                parent_nll=float(base[:,i].mean()), compressed_nll=float(arr[:,i].mean()),
                damage=float(means[i]), relative_damage=float(means[i]/base[:,i].mean()))
                for i,p in enumerate(PARENTS)})
        if 'gptq3' in table and 'gptq3-cal1' in table:
            for name, entry in table['gptq3']['contrasts'].items():
                diff = abs(entry['estimate'] - table['gptq3-cal1']['contrasts'][name]['estimate'])
                entry.update(calibration_difference=diff, calibration_flag=abs(entry['estimate']) < 2*diff)
        datasets[ds] = table
    agreement = {}
    for compression in ('pre-S1', 'rtn4', 'rtn3', 'gptq4', 'gptq3'):
        agreement[compression] = {}
        for contrast in ('L_ann', 'L_stab', 'R_short'):
            labels = {ds:table[compression]['contrasts'][contrast]['label']
                      for ds,table in datasets.items() if compression in table}
            agreement[compression][contrast] = dict(labels=labels,
                consistent=len(labels)==3 and len(set(labels.values()))==1)
    return dict(resamples=resamples, seed=seed, datasets=datasets, across_datasets=agreement,
                uncertainty='final-set sequences, not training or calibration seeds')


def pooled(tasks, resamples=10000, seed=0):
    from eval.a2_stage1_stats import MODELS, CONTRASTS as contrasts
    weights = np.array(list(contrasts.values()), dtype=float).T
    rng = np.random.default_rng(seed); task_points = {}; task_draws = {}
    for task in sorted(tasks['dclm']):
        # Joint categories preserve cross-corpus covariance and within-corpus pairing.
        matrix = np.stack([tasks[ds][task][m] for ds in ('dclm','fwe') for m in MODELS], axis=1)
        n = len(matrix)
        patterns, counts = np.unique(matrix, axis=0, return_counts=True)
        sampled = rng.multinomial(n, counts/n, size=resamples) @ patterns / n
        task_points[task] = np.stack([matrix[:, :6].mean(0) @ weights, matrix[:, 6:].mean(0) @ weights])
        task_draws[task] = np.stack([sampled[:, :6] @ weights, sampled[:, 6:] @ weights])
    reports = {}
    for metric, selected in [('mc_mean',[t for t in task_points if t!='lambada_openai']),
                             ('mc7_without_boolq',[t for t in task_points if t not in ('lambada_openai','boolq')]),
                             ('lambada_accuracy',['lambada_openai'])]:
        point = np.mean([task_points[t] for t in selected], axis=0)
        draws = np.mean([task_draws[t] for t in selected], axis=0)
        ci = np.quantile(draws.mean(0), [.025,.975], axis=0)
        reports[metric] = {name:dict(dclm=float(point[0,i]),fwe=float(point[1,i]),
            pooled=float(point[:,i].mean()),ci95=ci[:,i].tolist(),
            correlation=float(np.corrcoef(draws[0,:,i],draws[1,:,i])[0,1])
            if np.std(draws[0,:,i]) and np.std(draws[1,:,i]) else None)
            for i,name in enumerate(contrasts)}
    return dict(post_hoc=True,resamples=resamples,seed=seed,metrics=reports,
                sensitivity_reason='BoolQ parents at majority-class rate 62.2%',
                method='joint paired item-category resamples shared across both datasets')
