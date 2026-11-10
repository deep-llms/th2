"""Family v1.11 additions; previous experiment outputs remain immutable."""
import math
from pathlib import Path
import numpy as np
from eval.a2_family_f9 import read
from scripts.a2_quant_io import write as atomic_write

SPEC_SHA = '6ac9f6d2da0a75c1ac5d9846a00519de8750130813a19e10fc454bbd0490f22b'


def write(path, value):
    return atomic_write(path, dict(value, spec_sha256=SPEC_SHA))


def select_models(entries, plan):
    byid={e['model_id']:e for e in entries}
    if len(byid)!=55 or len(byid)!=len(entries):raise ValueError('Expected unique closed F9 inventory')
    parents=[byid[i] for i in plan['oneshots']]
    if len(parents)!=11 or any(e['role']!='parent' for e in parents):raise ValueError('Expected 11 parents')
    oneshots=[]
    for e in parents:
        ref=plan['oneshots'][e['model_id']]
        removed=list(range(24,33)) if e['family']=='smollm3' else list(range(16,22))
        if ref['removed_layers']!=removed:raise ValueError('Wrong registered main cut')
        oneshots.append(dict(e,model_id=e['model_id']+'-oneshot',parent_id=e['model_id'],role='oneshot',
                             oneshot=True,removed_layers=removed,oneshot_expected_nll=ref['expected_nll'],
                             final_nll=ref['expected_nll'],oneshot_reference=ref,budget=0.))
    original=[byid[i] for i in plan['new_task_original_ids']]
    if len(original)!=34 or len({e['model_id'] for e in original})!=34:raise ValueError('Expected 34 original models')
    if any(e['family']!='opensci' or e['dataset'] not in ('dclm','fwe') or e['arm'] not in ('P300','P1T','S72','S188') for e in original):
        raise ValueError('Unregistered new-task model')
    groups=[[e for e in original if e['role']=='parent'],[e for e in oneshots if e['family']=='opensci']]
    for dataset,arms in [('dclm',('P300','P1T')),('fwe',('P300','P1T')),(None,('S72','S188'))]:
        group=[e for e in original if e['role']=='student' and e['arm'] in arms and (dataset is None or e['dataset']==dataset)]
        groups.append(sorted(group,key=lambda e:(e['budget'],e['dataset'],arms.index(e['arm']))))
    if [len(g) for g in groups]!=[8,8,8,6,12]:raise ValueError('Wrong F10c model groups')
    return oneshots,groups


def holm(pvalues):
    """Step-down Holm adjusted p, preserving the original test order."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid p values')
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result.tolist()


def test_statistic(point, draws):
    d = np.asarray(draws, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all() or not math.isfinite(point):
        raise ValueError('Invalid bootstrap')
    se = float(d.std(ddof=1))
    z = point / se if se else None
    p = math.erfc(abs(z) / math.sqrt(2)) if se else (1. if point == 0 else 0.)
    # Center the empirical sampling distribution at H0=0; +1 avoids a zero
    # Monte Carlo p from finite resampling. Explicitly recorded in the report.
    boot = (1 + np.count_nonzero(np.abs(d - point) >= abs(point))) / (len(d) + 1)
    return dict(E_down_pp=float(point), bootstrap_SE_pp=se, z=z, p_normal=p,
                p_bootstrap_two_sided=float(boot),
                degenerate_SE=se == 0, ci95_pp=np.quantile(d, [.025, .975]).tolist())


def f10a(root, output):
    """Replay the exact F9 joint sampler, then correct the registered 90 tests."""
    from efficient_llm.diagnostic_data import sha256
    from eval.a2_family_f9_v110 import sampler
    from eval.downstream_suite import COUNTS
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest_path = root / 'backup/verified.json'
    expected = '8b0b68bdc69cf6aa7532250e1f368ae37fdcf1b6cebc7bdb099814b72633de3a'
    if sha256(manifest_path) != expected:
        raise ValueError('F9 backup manifest changed')
    manifest = read(manifest_path)
    def checked(name):
        p = root / name
        entry = manifest['files'][name]
        if p.stat().st_size != entry['bytes'] or sha256(p) != entry['sha256']:
            raise ValueError('Changed F9 artifact: ' + name)
        return read(p)
    report = checked('contrasts.json')
    records = {}
    for model in report['models']:
        saved = checked('per_item/' + model + '.json')
        records[model] = {}
        for task, rows in saved['tasks'].items():
            if len(rows) != COUNTS[task] or len({r['doc_id'] for r in rows}) != len(rows):
                raise ValueError('Incomplete saved item coverage')
            values = [r['correctness'] for r in rows]
            if not np.isin(values, [0, 1]).all():
                raise ValueError('Nonbinary accuracy')
            if abs(np.mean(values) - report['models'][model][task]) > 1e-12:
                raise ValueError('Saved aggregate does not match items')
            records[model][task] = dict(identity=saved['continuous_records'][task]['identity'], values=values)
    sample = sampler(records, resamples=10000, seed=0)
    families = {}
    for family, pair, cells in [
        ('annealed', ['P300', 'P1T'], [('pooled', b) for b in (.2, .8, 3.2)] + [('dclm', 12.8)]),
        ('stable', ['S72', 'S188'], [('pooled', b) for b in (.2, .8, 3.2)]),
        ('smollm3', ['k6', 'k20'], [('fwe', b) for b in (.2, .8, 3.2)]),
    ]:
        tests = []
        for dataset, budget in cells:
            candidates = [r for r in report['pooled'] + report['lead']
                          if r['pair'] == pair and r['dataset'] == dataset and r['budget'] == budget]
            if len(candidates) != 1:
                raise ValueError('Ambiguous registered contrast: ' + repr((family, dataset, budget)))
            row = candidates[0]
            names, draws = sample(row['models'])
            if len(names) not in (4, 8):
                raise ValueError('Unexpected contrast size')
            weights = np.tile([-1., 1., 1., -1.], len(names) // 4) / (len(names) // 4)
            for task in sorted(COUNTS):
                means, boots = draws[task]
                point, values = float(means @ weights * 100), boots @ weights * 100
                original = row['E_down'][task]
                if not np.allclose([point, *np.quantile(values, [.025, .975])],
                                   [original['estimate'], *original['ci95']], atol=1e-10, rtol=1e-10):
                    raise ValueError('F9 joint bootstrap replay mismatch')
                tests.append(dict(task=task, dataset=dataset, budget_B=budget,
                                  models=names, **test_statistic(point, values)))
        for row, adjusted in zip(tests, holm([r['p_normal'] for r in tests]), strict=True):
            row.update(p_holm=adjusted, survives=adjusted <= .05,
                       sign='positive' if row['E_down_pp'] > 0 else 'negative' if row['E_down_pp'] < 0 else 'zero')
        families[family] = tests
    if [len(v) for v in families.values()] != [36, 27, 27]:
        raise ValueError('Wrong multiplicity families')
    output.mkdir(parents=True)
    result = dict(families=families, resamples=10000, seed=0, alpha=.05,
                  retrospective=True, f9_manifest_sha256=expected,
                  bootstrap_p_definition='(1 + count(abs(E_boot-E_observed) >= abs(E_observed))) / (B+1); centered two-sided null',
                  standard_error_definition='sample standard deviation of bootstrap estimates, ddof=1',
                  correction='Holm separately within 36/27/27 tests on two-sided normal-approximation p',
                  all_F9_estimates_and_intervals_replayed=True)
    write(output / 'tests.json', result)
    import csv
    columns = ['family', 'task', 'dataset', 'budget_B', 'E_down_pp', 'bootstrap_SE_pp', 'z', 'p_normal', 'p_bootstrap_two_sided', 'p_holm', 'survives', 'sign']
    with (output / 'tests.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore'); writer.writeheader()
        for family, rows in families.items():
            writer.writerows(dict(row, family=family) for row in rows)
    write(output / 'complete.json', dict(complete=True, tests=90, survivors={k:sum(r['survives'] for r in v) for k,v in families.items()},
                                       files={n:sha256(output/n) for n in ('tests.json','tests.csv')}))
    print('F10A_COMPLETE', {k:sum(r['survives'] for r in v) for k,v in families.items()}, flush=True)
