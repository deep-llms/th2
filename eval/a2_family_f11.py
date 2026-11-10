"""Registered v1.13 identities and temperature calculations (no model runs)."""
from pathlib import Path
import math
import numpy as np
from scripts.a2_quant_io import write as atomic_write
from eval.a2_family_f9 import read

SPEC_SHA='a35e8f27d306bdc786426b1d57b6d075c4c9b7f6e437f4d95954b4dcc3409667'
BASE_GRID=list(range(70,131,2))
# Extra reductions on the SAME logits avoid a second model forward if the
# preregistered edge rule fires. They never enter selection unless triggered.
GUARD_GRID=list(range(60,141,2))

def write(path,value):
    return atomic_write(path,dict(value,spec_sha256=SPEC_SHA))

def choose_temperature(losses):
    if not all(math.isfinite(float(losses[str(t)])) for t in GUARD_GRID):
        raise ValueError('Nonfinite or missing temperature loss')
    initial=min(BASE_GRID,key=lambda t:losses[str(t)])
    grid=BASE_GRID.copy();extension=None
    if initial==70:grid=list(range(60,131,2));extension='lower'
    if initial==130:grid=list(range(70,141,2));extension='upper'
    chosen=min(grid,key=lambda t:losses[str(t)])
    return dict(T_star=chosen/100,T_key=str(chosen),initial_T=initial/100,
                extension=extension,active_grid=[t/100 for t in grid],
                still_at_edge=chosen in (grid[0],grid[-1]),
                extension_policy='one 0.1 extension on initial edge; flag if still at edge',
                guard_reductions='same forward; inactive values excluded from selection')

def lm_reading(E):
    if E>=.008:return 'persists'
    if abs(E)<.003:return 'erased'
    if E<=-.008:return 'reversed'
    return 'weak'

def calibration_reading(E,Ecal):
    if abs(E)<1e-15:return None,None
    r=Ecal/E
    return r,'mostly calibration' if r<.5 else 'not calibration' if r>=.75 else 'partly calibration'

def temperature_sums(logits,targets,grid=GUARD_GRID):
    """FP32 head logits as native score; accumulate each CE in float64."""
    import torch
    from torch.nn import functional as F
    logits=logits.float()
    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite logits')
    sums={}
    for t in grid:
        ce=F.cross_entropy(logits/(t/100),targets,reduction='none')
        sums[str(t)]=ce.double().sum()
    logp=logits.log_softmax(-1)
    entropy=-(logp.exp()*logp).sum(-1).double().sum()
    return sums,entropy

def document_bootstrap(matrix,counts,repeats=10000):
    """Joint document bootstrap, token-weighted exactly as native NLL."""
    matrix=np.asarray(matrix,float);counts=np.asarray(counts,float)
    if matrix.ndim!=2 or len(matrix)!=len(counts) or not len(counts) or np.any(counts<=0) or not np.isfinite(matrix).all():
        raise ValueError('Invalid paired documents')
    n=len(counts);rng=np.random.default_rng(0);draws=[]
    for start in range(0,repeats,64):
        ids=rng.integers(0,n,size=(min(64,repeats-start),n))
        draws.append((matrix[ids]*counts[ids][...,None]).sum(1)/counts[ids].sum(1)[:,None])
    return (matrix*counts[:,None]).sum(0)/counts.sum(),np.concatenate(draws)

def interval(point,draws):
    return dict(estimate=float(point),ci95=np.quantile(draws,[.025,.975]).tolist())
