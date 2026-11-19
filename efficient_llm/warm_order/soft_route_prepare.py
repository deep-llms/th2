"""Reserve within-seed new rows after the verified Horizon dense endpoint."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import Rows,sha
from .evaluate import cluster_ids
from .strength_refinement import check
from .train import write_json
from .soft_route import load_initial

NAMES=[f'{a}-s{s}' for a in ('C','G','S') for s in (0,1)]

def prepare(plan,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh preparation required')
    check(plan['hashes'])
    parent=json.loads(Path(plan['parent_report']).read_text())
    if parent['continuation_all_seeds_domains'] or parent['curriculum_all_seeds_domains']:
        raise ValueError('Resolve positive parent before changing architecture')
    previous=Path(plan['parent_inputs'])
    manifest=json.loads(previous.read_text())
    for rel,h in manifest['files'].items():
        if sha(previous.parent/rel)!=h:raise ValueError('Parent input changed')
    tc=plan['training']['0'];rows=Rows(tc['data'],tc['model']['length'])
    ids=np.load(plan['eval']['indices']);clusters=cluster_ids(rows,np.arange(len(rows.ids)))
    forbidden=np.unique(clusters[np.r_[ids['fit'],ids['valid']]])
    if np.intersect1d(clusters[ids['fit']],clusters[ids['valid']]).size:raise ValueError('Reserved document overlap')
    pool=rows.eligible();pool=pool[~np.isin(clusters[pool],forbidden)]
    out.mkdir(parents=True);record={}
    for seed in (0,1):
        prior=previous.parent/f'train-s{seed}.npy';old=np.load(prior)
        if manifest['files'].get(prior.name)!=sha(prior):raise ValueError('Unpinned parent order')
        allowed=np.setdiff1d(pool,old);t=plan['training'][str(seed)];needed=t['batch']*t['steps']
        if len(allowed)<needed:raise ValueError('Insufficient unused parent rows')
        chosen=np.random.default_rng(19190+seed).choice(allowed,needed,replace=False)
        np.save(out/f'train-s{seed}.npy',chosen)
        record[str(seed)]=dict(available=len(allowed),selected=needed,parent_row_overlap=int(np.intersect1d(old,chosen).size),reserved_doc_overlap=int(np.intersect1d(clusters[chosen],forbidden).size))
    for name in NAMES:
        arm,s=name.split('-');seed=int(s[1:])
        t=dict(plan['training'][str(seed)],arm=arm,seed=seed,indices=str(out/f'train-s{seed}.npy'),initial_weights=plan['weights'][str(seed)],parent_metadata=plan['parent_metadata'][str(seed)])
        t['expected_inputs']={p:sha(p) for p in (t['data']+'.bin',t['data']+'.starts',t['indices'],t['initial_weights'])}
        if arm=='C':
            model=load_initial(t);del model
            print('SOFT_ROUTE_PARENT_LOAD_PASS',seed,flush=True)
        write_json(out/f'{name}.json',dict(name=name,train=t,eval=plan['eval'],hashes={**plan['hashes'],**t['expected_inputs']}))
    write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],reservation=record,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('SOFT_ROUTE_INPUTS_READY',sha(out/'complete.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
