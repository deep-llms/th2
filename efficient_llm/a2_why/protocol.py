"""Fixed run matrix and decision clauses; no extra sweeps."""
import numpy as np

DATASETS = ('dclm', 'fwe')
ARMS = ('P300', 'P1T')
STEPS = {'0.2B':762, '0.8B':3051, '3.2B':12207}


def descriptor(phase, ds, arm, *, seed=2030, variant='drop', layers=6, budget='0.2B'):
    suffix = f'S1-0.2B-seed{seed}' if phase=='Y1' else f'{variant}-{budget}' if phase=='Y2' else f'L{layers}-{budget}'
    name = 'YT1-ST1-dclm-P300-S1-0.2B-lr1e-3' if phase=='YT1' else f'{phase}-{ds}-{arm}-{suffix}'
    return dict(run_id=name,phase=phase,dataset=ds,arm=arm,stream_seed=seed,variant=variant,
                removed=list(range(22-layers,22)),budget=budget,steps=STEPS[budget],lr=.001,seed=0,data_seed=0)


def base_runs():
    return [descriptor('YT1','dclm','P300')] + [
        descriptor('Y1',ds,arm,seed=seed) for seed in (2031,2032) for ds in DATASETS for arm in ARMS] + [
        descriptor('Y2',ds,arm,variant=v) for v in ('rdsc','avg') for ds in DATASETS for arm in ARMS] + [
        descriptor('Y3',ds,arm,layers=n,budget=b) for n in (3,9,12) for b in ('0.2B','0.8B') for ds in DATASETS for arm in ARMS]


def noise(es):
    """Sample std (ddof=1), fixed before new fits; population std also reported."""
    out={ds:dict(values=list(es[ds]),mean=float(np.mean(es[ds])),std=float(np.std(es[ds],ddof=1)),
        population_std=float(np.std(es[ds],ddof=0)),minimum=float(np.min(es[ds])),maximum=float(np.max(es[ds])),
        range=float(np.ptp(es[ds]))) for ds in DATASETS}
    sigma=float(np.mean([out[ds]['std'] for ds in DATASETS]))
    return dict(datasets=out,sigma=sigma,tau=max(2*sigma,.003),std_ddof=1)


def merge_reading(es,drop,tau):
    if all(es[d]<=drop[d]-tau for d in DATASETS):return 'B supported'
    if all(abs(es[d]-drop[d])<tau for d in DATASETS):return 'no effect'
    return 'mixed'


def capacity_reading(es,tau):
    if all(all(es[d][a]<es[d][b] for a,b in ((3,6),(6,9),(9,12))) for d in DATASETS):return 'A supported'
    if all(es[d][12]-es[d][3]<tau for d in DATASETS):return 'A not supported'
    return 'mixed'


def extend_capacity(e02,e08):
    if e02==0:return e08!=0
    return abs(e08-e02)/abs(e02)>.20


def spearman(es):
    values=np.asarray(es,dtype=float)
    ranks=np.array([np.sum(values<v)+(np.sum(values==v)+1)/2 for v in values])
    return None if np.std(ranks)==0 else float(np.corrcoef(np.arange(len(values)),ranks)[0,1])
