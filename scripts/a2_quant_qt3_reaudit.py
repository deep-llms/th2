"""CPU-only read-only audit of saved A2 Q weights under the user-revised QT3.

Never invokes RTN/GPTQ or model forwards; original artifacts remain immutable.
GPTQ grids were not saved. Its fallback solves for a single affine uniform grid
whose levels lie in the stored BF16 values' rounding intervals (plus one FP32
relative ulp for the original multiply). A feasible grid is a witness, not an
assertion that the historical grid/codes have been recovered.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def bf16_intervals(v):
    """Closed rounding cells; ties admitted at endpoints. Input exact BF16."""
    v=np.asarray(v,dtype=np.float32)
    bits=(v.view(np.uint32)>>16).astype(np.uint16)
    pos=v>0;neg=v<0
    below=np.where(pos,bits-1,np.where(neg,bits+1,0x8001)).astype(np.uint32)
    above=np.where(pos,bits+1,np.where(neg,bits-1,1)).astype(np.uint32)
    prev=(below<<16).view(np.float32).astype(np.float64)
    nxt=(above<<16).view(np.float32).astype(np.float64)
    x=v.astype(np.float64)
    allowance=np.maximum(np.abs(prev),np.abs(nxt))*2**-23
    return (prev+x)/2-allowance,(nxt+x)/2+allowance


def distinct(q,levels):
    ordered=np.sort(q,axis=1)
    changes=np.ones(ordered.shape,dtype=bool)
    changes[:,1:]=ordered[:,1:]!=ordered[:,:-1]
    count=changes.sum(1)
    u=np.zeros((len(q),levels),dtype=np.float32)
    rows,cols=np.nonzero(changes)
    rank=np.cumsum(changes,axis=1)[rows,cols]-1
    take=rank<levels;u[rows[take],rank[take]]=ordered[rows[take],cols[take]]
    return count,u


def uniform_bf16(u,count,bits):
    """Find an explicit feasible level-index assignment for every passed group.

    Enumerate endpoint code spans 1..2^b-1. Pairwise interval bounds on scale
    are necessary and sufficient for an offset to exist for a proposed code
    assignment. Failure to find a witness stays a failure; never silently pass.
    """
    levels=2**bits;ok=count==1
    low,high=bf16_intervals(u)
    for span in range(levels-1,0,-1):
        ix=np.flatnonzero(~ok & (count<=span+1))
        if not len(ix):continue
        vals=u[ix].astype(np.float64);n=count[ix];live=np.arange(levels)[None,:]<n[:,None]
        width=vals[np.arange(len(ix)),n-1]-vals[:,0]
        codes=np.rint((vals-vals[:,:1])/width[:,None]*span).astype(np.int32)
        valid=np.all((np.diff(codes,axis=1)>0)|~live[:,1:],axis=1)
        slo=np.zeros(len(ix));shi=np.full(len(ix),np.inf)
        lo=low[ix];hi=high[ix]
        for j in range(1,levels):
            gap=codes[:,j:j+1]-codes[:,:j]
            use=live[:,j:j+1]&live[:,:j]&(gap>0)
            safe=np.where(use,gap,1)
            lower=np.where(use,(lo[:,j:j+1]-hi[:,:j])/safe,-np.inf)
            upper=np.where(use,(hi[:,j:j+1]-lo[:,:j])/safe,np.inf)
            slo=np.maximum(slo,lower.max(1));shi=np.minimum(shi,upper.min(1))
        feasible=valid&(shi>=slo)&(shi>0)&np.isfinite(shi)
        ok[ix[feasible]]=True
    return ok


def inspect_matrix(q,parent,bits,chunk=4096):
    import torch
    levels=2**bits
    assert q.ndim==2 and q.shape[1]%128==0 and q.dtype==torch.bfloat16
    q=q.float().numpy().reshape(-1,128)
    p=None if parent is None else parent.bfloat16().float().numpy().reshape(-1,128)
    result=dict(groups=len(q),cardinality_failed_groups=0,grid_failed_groups=0,
                max_distinct_values=0,constant_groups=0,failures=[])
    for start in range(0,len(q),chunk):
        v=q[start:start+chunk];assert np.isfinite(v).all()
        count,u=distinct(v,levels);card=count<=levels
        if p is None:
            good=uniform_bf16(u,count,bits)&card
        else:
            w=p[start:start+chunk];lo=w.min(1,keepdims=True);hi=w.max(1,keepdims=True)
            scale=(hi-lo)/np.float32(levels-1);constant=scale==0
            safe=np.where(constant,np.float32(1),scale);zero=np.rint(-lo/safe)
            original=np.clip(np.rint(w/safe)+zero,0,levels-1)
            recovered=np.clip(np.rint(v/safe)+zero,0,levels-1)
            expected=np.where(constant,lo,(original-zero)*safe)
            expected=torch.from_numpy(expected).bfloat16().float().numpy()
            stable=np.where(constant,v==lo,recovered==original)
            good=stable.all(1)&(v==expected).all(1)&card
            result['constant_groups']+=int(constant.sum())
        result['cardinality_failed_groups']+=int((~card).sum())
        result['grid_failed_groups']+=int((~good).sum())
        result['max_distinct_values']=max(result['max_distinct_values'],int(count.max()))
        for j in np.flatnonzero(~good)[:max(0,4-len(result['failures']))]:
            result['failures'].append(dict(group=start+int(j),distinct=int(count[j]),values=np.unique(v[j]).tolist()))
    result['passed']=result['cardinality_failed_groups']==result['grid_failed_groups']==0
    return result


def worker(task):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    ds,parent,records,asset,out,spec=task
    parentpath=Path(asset['weights_dir'])/'model.safetensors'
    assert sha(parentpath)==asset['files']['model.safetensors']
    outputs=[]
    with safe_open(str(parentpath),framework='pt',device='cpu') as pf:
        for recordpath,expected_meta in records:
            assert sha(recordpath)==expected_meta
            meta=json.loads(Path(recordpath).read_text());wp=meta['weights']['path']
            assert sha(wp)==meta['weights']['sha256']
            compression=meta['compression'];bits=3 if compression.endswith('cal1') else int(compression[-1])
            rtn=compression.startswith('rtn');started=time.monotonic();matrices=[]
            with safe_open(wp,framework='pt',device='cpu') as f:
                assert set(f.keys())=={r['name'] for r in meta['matrices']}
                for row in meta['matrices']:
                    name=row['name'];q=f.get_tensor(name)
                    assert list(q.shape)==row['shape']
                    result=inspect_matrix(q,pf.get_tensor(name) if rtn else None,bits)
                    matrices.append(dict(name=name,shape=row['shape'],**result))
            assert len(matrices)==168
            item=dict(test='QT3_revised',dataset=ds,parent=parent,compression=compression,bits=bits,
                method='RTN parent grid: original codes == codes recovered from saved BF16 weights; exact saved values' if rtn else 'GPTQ grids not stored: uniform grid feasibility within BF16 rounding intervals',
                historical_grid_recovered=rtn,spec_sha256=spec,metadata=dict(path=recordpath,sha256=expected_meta),
                weights=meta['weights'],parent_weights=dict(path=str(parentpath),sha256=asset['files']['model.safetensors']),
                matrices=matrices,matrix_count=len(matrices),group_count=sum(r['groups'] for r in matrices),
                passed=all(r['passed'] for r in matrices),elapsed_seconds=time.monotonic()-started)
            path=Path(out)/(ds+'_'+parent+'_'+compression+'.json')
            path.write_text(json.dumps(item,indent=2)+'\n')
            summary={k:v for k,v in item.items() if k not in ('matrices','parent_weights','metadata','weights')}
            summary['receipt']=dict(path=str(path),sha256=sha(path))
            outputs.append(summary)
            print('QT3_REVISED_MODEL',json.dumps(summary),flush=True)
    return outputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--processes',type=int,default=12);a=p.parse_args()
    plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=False)
    tasks=[]
    for key,entry in plan['parents'].items():
        ds,parent=key.split('/');tasks.append((ds,parent,entry['records'],entry['asset'],str(a.output),plan['spec_sha256']))
    started=time.monotonic();models=[];errors=[]
    print('QT3_REVISED_START',json.dumps(dict(device='cpu',parents=len(tasks),models=56,processes=min(a.processes,len(tasks)),plan_sha256=sha(a.plan),code_sha256=sha(__file__))),flush=True)
    with ProcessPoolExecutor(max_workers=min(a.processes,len(tasks))) as pool:
        futures={pool.submit(worker,t):t[:2] for t in tasks}
        for future in as_completed(futures):
            try:models.extend(future.result())
            except Exception as error:errors.append(dict(parent=futures[future],error=repr(error)))
    result=dict(spec_sha256=plan['spec_sha256'],policy=plan['policy'],plan_sha256=sha(a.plan),
        audit_code_sha256=sha(__file__),device='cpu',new_quantization=False,models=models,errors=errors,
        expected_models=56,actual_models=len(models),matrices=sum(x['matrix_count'] for x in models),
        groups=sum(x['group_count'] for x in models),all_passed=len(models)==56 and not errors and all(x['passed'] for x in models),
        elapsed_seconds=time.monotonic()-started)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('QT3_REVISED_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='models'}),flush=True)

if __name__=='__main__':main()
