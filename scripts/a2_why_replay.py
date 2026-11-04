"""Independent CPU replay of exported Why per-sequence scores and LR schedules."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from efficient_llm.a2_why.common import read,write,SPEC_SHA


def replay(root):
    root=Path(root);checks=[]
    def check(name,ok,detail=None):checks.append(dict(test=name,passed=bool(ok),detail=detail))
    results=list(sorted(root.glob('heals/*/result.json')))
    check('results_present',bool(results))
    deltas={}
    for path in results:
        r=read(path);name=r['run_id'];item=r['registered_run'];steps=item['steps'];warmup=max(10,math.ceil(.02*steps))
        check(name+':spec',r['spec_sha256']==SPEC_SHA)
        expected=[.001*i/warmup if i<=warmup else .001*(steps-i)/(steps-warmup) for i in range(1,steps+1)]
        # Independent registered Stage1 linear warmup and linear decay to zero.
        actual=r['lr_updates'];check(name+':LR',len(actual)==steps and all(math.isclose(a,b,rel_tol=1e-12,abs_tol=0) for a,b in zip(actual,expected)))
        block=r['block'];vals={}
        for label in ('parent','pre-'+block,'post-'+block):
            p=root/'perseq'/name/(label+'.json')
            if not p.exists():check(name+':'+label,False,'missing');continue
            v=read(p);arr=v['per_sequence_nll'];valid=len(arr)==2461 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in arr)
            check(name+':'+label+':coverage',valid and v['tokens']==2461*4095)
            if valid:
                vals[label]=math.fsum(arr)/len(arr)
                check(name+':'+label+':mean',math.isclose(vals[label],v['loss'],rel_tol=0,abs_tol=1e-12))
        if 'parent' in vals and 'post-'+block in vals:
            delta=vals['post-'+block]-vals['parent'];deltas[name]=delta
            check(name+':delta',math.isclose(delta,r['delta_post'],rel_tol=0,abs_tol=2e-12))
        check(name+':roundtrip',r['roundtrip_difference'] is not None and r['roundtrip_difference']<=1e-6)
        order=read(path.parent/'order.json')['rank_microbatches'];joined=[]
        for step in range(len(order[0])):
            for rank in range(8):joined.extend(order[rank][step])
        check(name+':order',joined==list(range(steps*64)))
    return dict(status='PASS' if all(x['passed'] for x in checks) else 'FLAGGED',checks=checks,run_count=len(results),recomputed_delta=deltas,
                scope='Independent per-sequence NLL, deltas, LR and actual rank sample order; GPU correctness/resume not inferred')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();write(a.output,replay(a.root))
