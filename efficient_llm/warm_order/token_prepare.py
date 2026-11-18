"""CPU-only reservation and hash checks. No GPU management or label-based selection."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import Rows,sha
from .evaluate import cluster_ids
from .train import write_json


def reserve(rows,forbidden,fit_n=4096,valid_n=512):
    pool=np.setdiff1d(rows.eligible(),forbidden)
    fit=np.random.default_rng(55190).permutation(pool)[:fit_n]
    used=cluster_ids(rows,fit)
    rest=pool[~np.isin(cluster_ids(rows,pool),used)]
    valid=np.random.default_rng(55191).permutation(rest)[:valid_n]
    if len(fit)!=fit_n or len(valid)!=valid_n:raise ValueError('Insufficient fresh rows/documents')
    if np.intersect1d(fit,forbidden).size or np.intersect1d(valid,forbidden).size:raise ValueError('Row leakage')
    if np.intersect1d(cluster_ids(rows,fit),cluster_ids(rows,valid)).size:raise ValueError('Fit/validation document leakage')
    return fit,valid


def prepare(base,plan):
    base=Path(base);out=base/'inputs'
    if out.exists():raise ValueError('Prior inputs found; retrieve rather than overwrite')
    for p,want in plan['hashes'].items():
        if sha(p)!=want:raise ValueError('Input hash mismatch '+p)
    old=Path(plan['pilot']);local=Path(plan['local'])
    for study in (old,local):
        proof=json.loads((study/'inputs/complete.json').read_text())
        for rel,want in proof['files'].items():
            if sha(study/'inputs'/rel)!=want:raise ValueError('Prior prepared selection changed')
    excluded=[]
    for seed in (0,1):
        for study in (old,local):
            excluded += [np.load(study/f'inputs/train-s{seed}.npy'),np.load(study/f'inputs/eval-s{seed}.npz')['fit']]
    fit,valid=reserve(Rows(plan['stems']['fit']),np.unique(np.concatenate(excluded)))
    test=np.load(local/'inputs/eval-s0.npz',allow_pickle=False)
    out.mkdir(parents=True);np.savez(out/'indices.npz',fit=fit,valid=valid,dclm=test['dclm'],fwe=test['fwe'])
    for model in plan['models']:
        cfg={**model,'indices':str(out/'indices.npz'),'stems':plan['stems'],
             'hashes':{**plan['hashes'],str(out/'indices.npz'):sha(out/'indices.npz')}}
        write_json(out/(model['name']+'.json'),cfg)
    write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        counts=dict(fit=len(fit),valid=len(valid),dclm=len(test['dclm']),fwe=len(test['fwe'])),
        fit_valid_document_overlap=0,forbidden_rows=int(len(np.unique(np.concatenate(excluded))))))
    print('TOKEN_ORDER_INPUTS_READY',json.dumps(dict(sha256=sha(out/'complete.json'),proof=json.loads((out/'complete.json').read_text()))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    prepare(a.base,json.loads(Path(a.plan).read_text()))
if __name__=='__main__':main()
