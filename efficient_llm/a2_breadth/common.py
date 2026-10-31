"""Pure protocol/readout helpers; scientific tests are records, never gates."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

SPEC_SHA='dd24ef489e2fb3cb4f6e16fd7db0df4b412e96679786315cf44bb3229c88fd0b'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(obj,dict):obj=dict(obj,spec_sha256=SPEC_SHA)
    with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False,prefix=p.name+'.') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');tmp=f.name
    os.replace(tmp,p)
    return dict(path=str(p),sha256=sha(p))
def blocks(n):
    return dict(B_early=list(range(n//12,n//3)),B_S1=list(range(2*n//3,11*n//12)),B_end=list(range(3*n//4,n)))
class Checks:
    def __init__(self,path):self.path=Path(path);self.items=[]
    def record(self,name,passed,detail=None):
        e=dict(test=name,passed=bool(passed),detail=detail,spec_sha256=SPEC_SHA)
        self.items.append(e);write(self.path,dict(checks=self.items))
        print('BREADTH_CHECK',json.dumps(e),flush=True)
        return passed
    def attempt(self,name,fn):
        try:return fn()
        except Exception as e:
            self.record(name,False,dict(error=repr(e),action='flag and continue independent registered work'));return None

def weighted(nll,counts):
    nll=np.asarray(nll,dtype=float);counts=np.asarray(counts,dtype=float)
    return float(np.dot(nll,counts)/counts.sum())
def relative_bootstrap(early,late,counts,repeats=10000,seed=0):
    """Document-paired resampling; recompute token-weighted damage each draw."""
    early=np.asarray(early,dtype=float);late=np.asarray(late,dtype=float);counts=np.asarray(counts,dtype=float)
    rng=np.random.default_rng(seed);draws=[];undefined=0
    for _ in range(repeats):
        ix=rng.integers(0,len(early),len(early));den=float(np.dot(early[ix],counts[ix]))
        if abs(den)<1e-15:undefined+=1;continue
        draws.append(float(np.dot((late-early)[ix],counts[ix])/den))
    den=weighted(early,counts)
    return dict(value=(weighted(late,counts)-den)/den if abs(den)>1e-15 else None,
        ci95=np.quantile(draws,[.025,.975]).tolist() if draws else None,undefined_draws=undefined,
        resamples=repeats,seed=seed,unit='documents',estimator='token-weighted NLL')
def spearman(x,y):
    def rank(v):
        v=np.asarray(v);return np.array([np.sum(v<a)+(np.sum(v==a)-1)/2 for a in v])
    a,b=rank(x),rank(y)
    return float(np.corrcoef(a,b)[0,1]) if a.std() and b.std() else None
