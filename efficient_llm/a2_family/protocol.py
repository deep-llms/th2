"""Pure registered selection rules; no final-set selection of learning rate."""
import math

STEPS = {'0.2B': 762, '0.8B': 3051, '3.2B': 12207}

def removed_layers(n):
    return list(range(2*n//3, 11*n//12))

def descriptor(phase, arm, budget, lr, seed=2030, data="fwe"):
    if data not in ("fwe", "mix"):raise ValueError("Unregistered data")
    if arm not in ('k6', 'k20', 'k64') or seed not in (2030, 2031):
        raise ValueError('Unregistered arm or data order')
    return dict(run_id=f'{phase}-{data}-{arm}-BS1-{budget}-lr{lr:.8g}-s{seed}',
                phase=phase, data=data, arm=arm, budget=budget, steps=STEPS[budget],
                tokens=STEPS[budget]*64*4096, lr=lr, stream_seed=seed, model_seed=0)

def grid(peak=2e-4, extension=False):
    return [descriptor('F2', arm, '0.2B', peak/divisor)
            for divisor in ((16,) if extension else (8, 4, 2)) for arm in ('k6','k20')]

def choose_lr(rows, peak=2e-4):
    # Missing/nonfinite results are explicit, never silently declared optimal.
    scores = {}
    missing = []
    for divisor in (16, 8, 4, 2):
        lr = peak/divisor
        values = [rows.get((arm, lr)) for arm in ('k6', 'k20')]
        if all(v is not None and math.isfinite(v) for v in values):
            scores[lr] = sum(values)/2
        elif divisor != 16 or any(v is not None for v in values):
            missing.append(lr)
    if not scores:
        return dict(lr=peak/4, unverified=True, reason='No complete finite tuning pair; registered matched LR fallback', missing=missing)
    selected = min(scores, key=lambda lr: (scores[lr], lr))
    return dict(lr=selected, tuning_means=scores, missing=missing,
                unverified=bool(missing), extend_down=(selected==peak/8 and peak/16 not in scores),
                upper_edge=(selected==peak/2))

def changed(previous, current):
    return abs(current-previous) > max(.2*abs(previous), .003)

def extension_arms(e02, e08, d02, d08):
    f3 = changed(e02, e08)
    f4 = f3 or changed(d02, d08)
    return dict(F3=['k6','k20'] if f3 else [],
                F4=(['k64'] if f3 else ['k64','k20']) if f4 else [])

def readings(e, e02, e08, d, previous_e, previous_d):
    return dict(FE1='persists' if e>=.008 else 'erased' if abs(e)<.003 else 'reversed' if e<=-.008 else 'weak',
                FE2='flat' if abs(e08-e02)<.003 else 'changing_up' if e08>e02 else 'changing_down',
                FE3='grows' if d>=.003 else 'saturates' if abs(d)<.003 else 'falls',
                FE1_not_converged=changed(previous_e,e), FE3_not_converged=changed(previous_d,d))

def break_even(budgets, nll, reference):
    pairs = sorted(zip(budgets,nll))
    for i,(budget,value) in enumerate(pairs):
        if value <= reference:
            if i==0:return dict(budget=budget, bracket=None, interpolation=False)
            b0,v0=pairs[i-1]
            fraction=(v0-reference)/(v0-value)
            return dict(budget=math.exp(math.log(b0)+fraction*math.log(budget/b0)), bracket=[b0,budget], interpolation=True)
    return dict(budget=None, largest_budget=pairs[-1][0], gap=pairs[-1][1]-reference)


def mixture_reading(e_mix,e_fwe,e_second_order):
    difference=e_mix-e_fwe
    dependent=abs(difference)>=.003
    return dict(FE4='data-dependent' if dependent else 'robust',difference=difference,
                direction='up' if difference>0 else 'down' if difference<0 else 'equal',
                data_order_difference=abs(e_second_order-e_fwe),run_F5=dependent)
