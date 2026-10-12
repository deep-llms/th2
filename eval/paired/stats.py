"""Paired bootstrap and sign-flip statistics with explicit sampling units.

Cluster resampling keeps all questions in each sampled story. Point estimates
remain question-weighted, not an unannounced average of story averages.
"""
import itertools
import math
import numpy as np


def vector(x, minimum=2):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < minimum or not np.isfinite(x).all():
        raise ValueError('Expected finite vector with enough observations')
    return x


def units(n, clusters=None):
    if clusters is None:
        return [np.array([i]) for i in range(n)]
    if len(clusters) != n:
        raise ValueError('Cluster IDs must match observations')
    grouped = {}
    for i, key in enumerate(clusters):
        grouped.setdefault(str(key), []).append(i)
    if len(grouped) < 2:
        raise ValueError('Need at least two independent sampling units')
    return [np.asarray(v) for v in grouped.values()]


def effect(d):
    d = vector(d)
    if np.all(d==d[0]):
        if d[0]!=0: raise ValueError('standardized effect undefined: constant nonzero differences')
        return 0.
    sd = float(d.std(ddof=1))
    if sd == 0 and np.any(d != 0):
        raise ValueError('standardized effect undefined: constant nonzero differences')
    return float(d.mean() / sd) if sd else 0.


def _z(mu, sd):
    # An individual bootstrap/permutation replicate can be degenerate even
    # when the observed sample is valid. Do not turn its nonzero effect into 0.
    out = np.zeros_like(mu)
    np.divide(mu, sd, out=out, where=sd > 0)
    out[(sd == 0) & (mu > 0)] = np.inf
    out[(sd == 0) & (mu < 0)] = -np.inf
    return out


def _moments(s, q, n):
    mu = s / n
    sd = np.sqrt(np.maximum(q - s*s/n, 0.) / (n-1))
    return mu, _z(mu, sd)


def _settings(B, chunk):
    if type(B) is not int or B < 2 or type(chunk) is not int or chunk < 1:
        raise ValueError('Positive resampling settings required (B >= 2)')


def replicates(d, B=10000, seed=0, clusters=None, chunk=32):
    d = vector(d); effect(d); _settings(B, chunk)
    u = units(len(d), clusters)
    s = np.array([d[i].sum() for i in u]); q = np.array([np.square(d[i]).sum() for i in u])
    n = np.array([len(i) for i in u]); U = len(u)
    chunk = min(chunk, max(1, 1000000 // U))
    rng = np.random.default_rng(seed)
    bm = np.empty(B); bz = np.empty(B); fm = np.empty(B); fz = np.empty(B)
    for start in range(0, B, chunk):
        size = min(chunk, B-start)
        idx = rng.integers(U, size=(size,U))
        bm[start:start+size], bz[start:start+size] = _moments(s[idx].sum(1), q[idx].sum(1), n[idx].sum(1))
        signs = 2*rng.integers(2, size=(size,U), dtype=np.int8)-1
        fm[start:start+size], fz[start:start+size] = _moments((signs*s).sum(1), q.sum(), n.sum())
    return bm, bz, fm, fz


def interval(samples):
    # Infinite resample effects indicate an unbounded/undefined standardized
    # interval. JSON uses null plus an explicit availability flag, never NaN.
    if not np.isfinite(samples).all(): return None
    return np.quantile(samples,[.025,.975]).tolist()


def pvalue(null, observed, exact=False):
    if not math.isfinite(observed) or np.isnan(null).any():
        raise ValueError('Invalid test statistic')
    hits = int(np.count_nonzero(np.abs(null) >= abs(observed)*(1-1e-12)))
    return hits/len(null) if exact else (hits+1)/(len(null)+1)


def paired(d, B=10000, seed=0, clusters=None):
    d = vector(d); z = effect(d)
    bm,bz,fm,fz = replicates(d,B,seed,clusters)
    u = units(len(d),clusters)
    exact = len(u) <= 12
    if exact:
        sums = np.array([d[i].sum() for i in u])
        signs = np.array(list(itertools.product((-1.,1.),repeat=len(u))))
        fm,fz = _moments(signs@sums, np.square(d).sum(),len(d))
    return dict(N=len(d), units=len(u), mean=float(d.mean()), ci=interval(bm),
        std_effect=z, std_ci=interval(bz), std_ci_available=interval(bz) is not None,
        p=pvalue(fm,float(d.mean()),exact), p_std=pvalue(fz,z,exact), exact=exact,
        frac_pos=float(np.mean(d>0)), frac_neg=float(np.mean(d<0)))


def aggregate(ds, B=10000, seed=0, clusters=None):
    if not ds: raise ValueError('Cannot aggregate an empty task set')
    groups = clusters if clusters is not None else [None]*len(ds)
    if len(groups) != len(ds): raise ValueError('Task cluster count differs')
    reps = [replicates(d,B,seed+i,g) for i,(d,g) in enumerate(zip(ds,groups))]
    with np.errstate(invalid='ignore'):
        bm,bz,fm,fz = [np.mean([r[j] for r in reps],axis=0) for j in range(4)]
    mean = float(np.mean([vector(d).mean() for d in ds]))
    z = float(np.mean([effect(d) for d in ds]))
    if np.isnan(fz).any(): raise ValueError('Degenerate aggregate sign-flip statistic')
    return dict(mean=mean,ci=interval(bm),std_effect=z,std_ci=interval(bz),
                std_ci_available=interval(bz) is not None,p=pvalue(fm,mean),p_std=pvalue(fz,z))


def ratio(s, c, B=10000, seed=0):
    """Document-weighted ratio of sums, paired by document."""
    s,c = vector(s),vector(c)
    if s.shape != c.shape or np.any(c<=0): raise ValueError('Positive matched denominators required')
    _settings(B,32); rng=np.random.default_rng(seed); N=len(s)
    bt=np.empty(B); fl=np.empty(B)
    for start in range(0,B,32):
        k=min(32,B-start);idx=rng.integers(N,size=(k,N))
        bt[start:start+k]=s[idx].sum(1)/c[idx].sum(1)
        fl[start:start+k]=((2*rng.integers(2,size=(k,N))-1)*s).sum(1)/c.sum()
    est=float(s.sum()/c.sum())
    return dict(estimate=est,ci=interval(bt),p=pvalue(fl,est),N=N)


def joint_replicates(x, y, B, rng, clusters=None):
    """Resample aligned contrasts jointly without materializing question rows."""
    x,y=vector(x),vector(y)
    if x.shape!=y.shape: raise ValueError('Unpaired training-progress reference')
    u=units(len(x),clusters); U=len(u)
    counts=np.array([len(i) for i in u])
    sums=[np.array([d[i].sum() for i in u]) for d in (x,y)]
    squares=[np.array([np.square(d[i]).sum() for i in u]) for d in (x,y)]
    out=np.empty((2,2,B)); chunk=min(32,max(1,1000000//U))
    for start in range(0,B,chunk):
        size=min(chunk,B-start); idx=rng.integers(U,size=(size,U)); n=counts[idx].sum(1)
        for j in range(2):
            out[j,:,start:start+size]=_moments(sums[j][idx].sum(1),squares[j][idx].sum(1),n)
    return out


def joint_contrast(pairs, B=10000, seed=0, clusters=None, standardized=True):
    """(P_task2-P_task1)-(reference_task2-reference_task1), sharing A20."""
    if len(pairs)!=2: raise ValueError('A contrast requires two tasks')
    groups=clusters if clusters is not None else [None,None]
    if len(groups)!=2: raise ValueError('Task cluster count differs')
    _settings(B,32); rng=np.random.default_rng(seed); out=[]; estimates=[]
    fn=effect if standardized else lambda a:float(vector(a).mean())
    for (x,y),g in zip(pairs,groups):
        estimates.append(fn(x)-fn(y))
        reps=joint_replicates(x,y,B,rng,g)
        with np.errstate(invalid='ignore'):
            out.append(reps[0,int(standardized)]-reps[1,int(standardized)])
    with np.errstate(invalid='ignore'): ci=interval(out[1]-out[0])
    return dict(contrast=estimates[1]-estimates[0],ci=ci,
                interpretation='direct difference of contrasts, shared reference resampled jointly')


def contrast(ds, B=10000, seed=0, clusters=None):
    if len(ds)!=2: raise ValueError('A contrast requires two tasks')
    groups=clusters if clusters is not None else [None,None]
    if len(groups)!=2: raise ValueError('Task cluster count differs')
    reps=[replicates(d,B,seed+i,g) for i,(d,g) in enumerate(zip(ds,groups))]
    with np.errstate(invalid='ignore'): ci=interval(reps[1][1]-reps[0][1])
    return dict(contrast=effect(ds[1])-effect(ds[0]),ci=ci)


def progress_ratio(pairs, B=10000, seed=0, clusters=None):
    """Ratio of equal-task means; jointly resample shared A20 observations."""
    if not pairs: raise ValueError('Empty progress comparison')
    groups=clusters if clusters is not None else [None]*len(pairs)
    if len(groups)!=len(pairs): raise ValueError('Task cluster count differs')
    _settings(B,32); rng=np.random.default_rng(seed); num=np.zeros(B);den=np.zeros(B)
    means=[]
    for (x,y),g in zip(pairs,groups):
        x,y=vector(x),vector(y)
        reps=joint_replicates(x,y,B,rng,g)
        means.append((float(x.mean()),float(y.mean())))
        num+=reps[0,0]/len(pairs);den+=reps[1,0]/len(pairs)
    numerator,denominator=np.mean(means,axis=0);dci=interval(den)
    stable=dci is not None and not dci[0]<=0<=dci[1] and denominator!=0 and np.all(den!=0)
    return dict(numerator=float(numerator),denominator=float(denominator),denominator_ci=dci,
        estimable=bool(stable),ratio=float(numerator/denominator) if stable else None,
        ci=interval(num/den) if stable else None,
        interpretation='descriptive fraction of observed training progress, not a PPL prediction')
