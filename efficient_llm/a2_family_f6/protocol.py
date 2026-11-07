"""Family v1.4 F6 registrations and conditional extension; CPU-only."""
import math

STEPS = {'0.2B': 762, '0.8B': 3051, '3.2B': 12207}


def descriptor(dataset, arm, budget):
    if dataset not in ('dclm', 'fwe') or arm not in ('S72', 'S188'):
        raise ValueError('Unregistered F6 checkpoint')
    if budget == '0.2B' and (dataset, arm) != ('dclm', 'S72'):
        raise ValueError('FT9 is one DCLM S72 run only')
    steps = STEPS[budget]
    return dict(run_id=f'ST1-{dataset}-{arm}-S1-{budget}-lr1e-3',
                dataset=dataset, arm=arm, budget=budget, steps=steps,
                tokens=steps*64*4096, lr=.001, stream_seed=2030,
                seed=0, removed=list(range(16, 22)), variant='drop',
                phase='FT9' if budget == '0.2B' else 'F6')


def initial_runs():
    return [descriptor('dclm', 'S72', '0.2B')] + [
        descriptor(ds, arm, '0.8B')
        for ds in ('dclm', 'fwe') for arm in ('S72', 'S188')]


def select_extension(e02, e08):
    rows = {}
    for ds in ('dclm', 'fwe'):
        old, new = e02.get(ds), e08.get(ds)
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (old, new)):
            rows[ds] = dict(trigger=None, reason='Missing finite paired E; do not invent a trigger')
            continue
        difference = new-old
        rows[ds] = dict(E02=old, E08=new, difference=difference,
                        threshold=max(.2*abs(old), .003),
                        relative_change=abs(difference)/abs(old) if old else None,
                        trigger=abs(difference)>max(.2*abs(old), .003))
    fired = [ds for ds, row in rows.items() if row['trigger']]
    # Preregistered E02 is nonzero. Exact ties use stable DCLM-first order.
    chosen = max(fired, key=lambda ds: rows[ds]['relative_change']) if fired else None
    return dict(dataset=chosen, datasets=rows, tie_policy='DCLM first for an exact relative-change tie')


def reading(old, new):
    delta = new-old
    return dict(label='falls' if delta<=-.003 else 'rises' if delta>=.003 else 'flat', difference=delta)


def joint_reading(fe6, smol_fe2):
    down = smol_fe2.get('label') == 'changing' and smol_fe2.get('direction') == 'down'
    labels = [fe6.get(ds, {}).get('label') for ds in ('dclm', 'fwe')]
    if down and labels == ['falls', 'falls']:
        return 'stable-state effect'
    if down and labels == ['flat', 'flat']:
        return 'family difference'
    return 'mixed'
