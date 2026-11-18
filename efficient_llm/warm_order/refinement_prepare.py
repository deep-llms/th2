"""Reserve fresh fit windows and preferably untouched final documents, without scores."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .data import Rows,sha
from .model import Config
from .train import write_json
from .evaluate import cluster_ids


def select(rows,excluded,train_n,fit_n,valid_n):
    pool=np.setdiff1d(rows.eligible(),excluded);order=np.random.default_rng(66190).permutation(pool)
    train,fit=order[:train_n],order[train_n:train_n+fit_n]
    rest=order[train_n+fit_n:];rest=rest[~np.isin(cluster_ids(rows,rest),cluster_ids(rows,fit))]
    valid=rest[:valid_n]
    if (len(train),len(fit),len(valid))!=(train_n,fit_n,valid_n):raise ValueError('Insufficient fresh fit rows/documents')
    if len(np.unique(np.concatenate([train,fit,valid])))!=train_n+fit_n+valid_n:raise ValueError('Split leakage')
    return train,fit,valid


def prepare(plan,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh inputs required')
    for p,h in plan['hashes'].items():
        if sha(p)!=h:raise ValueError('Input changed '+p)
    pilot,local,token=[Path(plan[k]) for k in ('pilot','local','token')]
    for base in (pilot,local,token):
        proof=json.loads((base/'inputs/complete.json').read_text())
        for rel,h in proof['files'].items():
            if sha(base/'inputs'/rel)!=h:raise ValueError('Prior selection changed')
    excluded=[];oldtests=[]
    for seed in (0,1):
        for base in (pilot,local):
            excluded += [np.load(base/f'inputs/train-s{seed}.npy'),np.load(base/f'inputs/eval-s{seed}.npz')['fit']]
        oldtests.append(np.load(pilot/f'inputs/eval-s{seed}.npz')['test'])
    previous=np.load(token/'inputs/indices.npz');excluded += [previous['fit'],previous['valid']]
    excluded=np.unique(np.concatenate(excluded));steps=plan.get('steps',2000);batch=plan.get('batch',32)
    train,fit,valid=select(Rows(plan['stems']['fit']),excluded,steps*batch,4096,512)
    tests={};fresh={}
    for domain in ('dclm','fwe'):
        rows=Rows(plan['stems'][domain]);used=np.concatenate([previous[domain],*(oldtests if domain=='dclm' else [])]);pool=rows.eligible()
        pool=pool[~np.isin(cluster_ids(rows,pool),cluster_ids(rows,used))]
        fresh[domain]=len(pool)>=1024
        tests[domain]=np.random.default_rng(66192).permutation(pool)[:4096] if fresh[domain] else previous[domain]
    out.mkdir(parents=True);np.savez(out/'eval.npz',fit=fit,valid=valid,**tests)
    for seed in (0,1):
        np.save(out/f'train-s{seed}.npy',np.random.default_rng(66193+seed).permutation(train))
        for arm in ('C6','F7','J7'):
            data=plan['stems']['fit'];initial=plan['weights'][str(seed)];indices=str(out/f'train-s{seed}.npy')
            expected={p:plan['hashes'][p] for p in (data+'.bin',data+'.starts',initial)};expected[indices]=sha(indices)
            c=dict(arm=arm,seed=seed,model=asdict(Config()),steps=steps,batch=batch,lr=.0001,warmup=50,save_every=1000,log_every=50,threads=4,data=data,initial_weights=initial,indices=indices,expected_inputs=expected)
            write_json(out/f'{arm}-s{seed}.json',c)
    write_json(out/'eval-config.json',dict(stems=plan['stems'],indices=str(out/'eval.npz'),hashes={**plan['hashes'],str(out/'eval.npz'):sha(out/'eval.npz')},final_fresh_documents=fresh))
    write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()},counts=dict(train=len(train),fit=len(fit),valid=len(valid),**{k:len(v) for k,v in tests.items()}),forbidden_rows=len(excluded),fit_validation_document_overlap=0,final_fresh_documents=fresh,flags=['training_rows_fresh_but_documents_may_recur','two_seeds','equal_tokens_not_equal_compute']))
    print('REFINEMENT_INPUTS_READY',sha(out/'complete.json'),(out/'complete.json').read_text(),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
