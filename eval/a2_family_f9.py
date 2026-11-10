"""Family v1.9 shared identity and item-level statistical helpers (CPU only)."""
import json
import math
from pathlib import Path
import numpy as np
from efficient_llm.diagnostic_data import sha256
from eval.downstream_suite import COUNTS, PRIMARY
from eval.a2_request_bridge import digest
from scripts.a2_quant_io import write as atomic_write

SPEC_SHA='c25a43d4c074be0ae667f59d4b85e719adc73a357603c62a907bfd4d431e4085'

def read(path):
    return json.loads(Path(path).read_text())

def write(path,value):
    return atomic_write(path,dict(value,spec_sha256=SPEC_SHA))

def items(replay):
    result={}
    for task,n in COUNTS.items():
        rows=sorted(replay['samples'][task],key=lambda r:r['doc_id'])
        if len(rows)!=n or len({r['doc_id'] for r in rows})!=n:
            raise ValueError('Incomplete item coverage: '+task)
        values=np.asarray([r[PRIMARY[task].split(',')[0]] for r in rows],dtype=float)
        if not np.isin(values,[0,1]).all():raise ValueError('Nonbinary accuracy')
        if abs(values.mean()-replay['results'][task][PRIMARY[task]])>1e-12:
            raise ValueError('Harness aggregate mismatch')
        result[task]=dict(values=values.tolist(),identity=digest([
            (r['doc_id'],r['doc'],r['target'],r['arguments']) for r in rows]))
    return result

def batch_decision(first,second):
    a,b=items(first),items(second)
    differences={}
    for t in COUNTS:
        if a[t]['identity']!=b[t]['identity']:raise ValueError('Batch prompt identity changed')
        differences[t]=float(np.mean(b[t]['values'])-np.mean(a[t]['values']))
    mc=sum(v for t,v in differences.items() if t!='lambada_openai')/8
    la=differences['lambada_openai']
    passed=abs(mc)<=.001 and abs(la)<=.001
    return dict(passed=passed,batch_size=2 if passed else 1,
                MC_difference_pp=100*mc,LAMBADA_difference_pp=100*la,
                task_differences_pp={t:100*v for t,v in differences.items()})

def joint_draws(records,*,resamples=10000,seed=0):
    """Joint item bootstrap across ALL supplied models, including both corpora."""
    names=list(records);rng=np.random.default_rng(seed);out={}
    for task in sorted(COUNTS):
        rows=[records[n][task] for n in names]
        if len({r['identity'] for r in rows})!=1:raise ValueError('Unpaired task items')
        matrix=np.asarray([r['values'] for r in rows],dtype=float).T
        if matrix.shape[0]!=COUNTS[task]:raise ValueError('Missing task items')
        patterns,counts=np.unique(matrix,axis=0,return_counts=True)
        n=len(matrix)
        chunks=[]
        for start in range(0,resamples,128):
            chunks.append(rng.multinomial(n,counts/n,size=min(128,resamples-start))@patterns/n)
        out[task]=(matrix.mean(0),np.concatenate(chunks))
    for label,ts in [('MC8',[t for t in COUNTS if t!='lambada_openai']),
                     ('MC7',[t for t in COUNTS if t not in ('lambada_openai','boolq')])]:
        out[label]=(sum(out[t][0] for t in ts)/len(ts),sum(out[t][1] for t in ts)/len(ts))
    return names,out

def interval(point,draws,*,scale=100):
    lo,hi=np.quantile(draws,[.025,.975],method='linear')
    return dict(estimate=float(point*scale),ci95=[float(lo*scale),float(hi*scale)])

def pair_report(records,short_parent,long_parent,short_student,long_student):
    names,draws=joint_draws(records)
    ix=[names.index(n) for n in (short_parent,long_parent,short_student,long_student)]
    out={}
    for task,(p,d) in draws.items():
        sp,lp,ss,ls=ix
        gap=interval(p[lp]-p[sp],d[:,lp]-d[:,sp])
        e=interval(p[lp]-p[ls]-p[sp]+p[ss],d[:,lp]-d[:,ls]-d[:,sp]+d[:,ss])
        kept=None
        if gap['ci95'][0]>0 or gap['ci95'][1]<0:
            den=d[:,lp]-d[:,sp]
            # The requested CI gate is not permission to invent finite ratios.
            if np.all(den!=0):
                kept=interval((p[ls]-p[ss])/(p[lp]-p[sp]),(d[:,ls]-d[:,ss])/den,scale=1)
        out[task]=dict(E_down=e,parent_gap=gap,accuracy_kept_share=kept,
                       reading='detected' if e['ci95'][0]>0 or e['ci95'][1]<0 else 'not detected')
    return out
