"""Authenticated ten-parent recovery reservation and configuration."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import Rows, sha
from .evaluate import cluster_ids
from .strength_refinement import check
from .train import write_json
from .exposure_recovery import load_parent

NAMES = [f'{a}-s{s}' for a in ('C','G','S','GF','SF') for s in (0,1)]


def prepare(plan, out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh reservation required')
    check(plan['hashes'])
    if set(plan['parents']) != set(NAMES):raise ValueError('Ten parents required')
    t=plan['training'];rows=Rows(t['data'],t['model']['length'])
    clusters=cluster_ids(rows,np.arange(len(rows.ids)));forbidden=[]
    for item in plan['reserved']:
        if plan['hashes'].get(item['path']) != sha(item['path']):raise ValueError('Unpinned reservation')
        data=np.load(item['path'],allow_pickle=False)
        ids=np.concatenate([data[k] for k in item['keys']]) if item['keys'] else data
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or ids.min()<0 or ids.max()>=len(rows.ids):raise ValueError('Invalid reserved rows')
        forbidden.extend(clusters[ids].tolist())
    forbidden=np.unique(forbidden)
    pool=rows.eligible();pool=pool[~np.isin(clusters[pool],forbidden)]
    out.mkdir(parents=True);record={}
    for seed in (0,1):
        old=[]
        for p in plan['excluded_orders'][str(seed)]:
            if plan['hashes'].get(p)!=sha(p):raise ValueError('Unpinned previous order')
            old.extend(np.load(p,allow_pickle=False).tolist())
        old=np.unique(old);allowed=np.setdiff1d(pool,old);needed=t['steps']*t['batch']
        if len(allowed)<needed:raise ValueError('Insufficient new rows')
        chosen=np.random.default_rng(19201+seed).choice(allowed,needed,replace=False)
        np.save(out/f'train-s{seed}.npy',chosen)
        record[str(seed)]=dict(available=len(allowed),selected=needed,
            prior_row_overlap=int(np.intersect1d(old,chosen).size),
            reserved_doc_overlap=int(np.intersect1d(clusters[chosen],forbidden).size))
    for name in NAMES:
        p=plan['parents'][name];seed=int(name[-1]);parent=Path(p['directory'])
        hashes=p['hashes']
        for f,h in hashes.items():
            if plan['hashes'].get(str(parent/f))!=h:raise ValueError('Parent missing from plan pins')
        m,opt,state=load_parent(parent,hashes,name,expected_steps=plan['parent_steps'])
        if state['recipe']['config']['model']!=t['model'] or state['recipe']['config']['batch']!=t['batch']:
            raise ValueError('Parent model/batch changed')
        del m,opt,state
        c=dict(t,name=name,seed=seed,arm=name[0],parent=str(parent),parent_hashes=hashes,
            parent_steps=plan['parent_steps'],indices=str(out/f'train-s{seed}.npy'),
            schedule='inherit-terminal-lr-constant-no-floor')
        c['expected_inputs']={p:sha(p) for p in (c['data']+'.bin',c['data']+'.starts',c['indices'])}
        write_json(out/f'{name}.json',dict(name=name,train=c,eval=plan['eval'],hashes={**plan['hashes'],**c['expected_inputs']}))
        print('RECOVERY_PARENT_LOAD_PASS',name,flush=True)
    write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],reservation=record,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('EXPOSURE_RECOVERY_INPUTS_READY',sha(out/'complete.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
