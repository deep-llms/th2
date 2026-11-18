"""Common scratch reservations excluding entire router fit/validation documents."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import Rows,sha
from .evaluate import cluster_ids
from .strength_refinement import check
from .train import write_json
from .progressive import path_for
NAMES=[f'{a}-s{s}' for a in ('D','R','WS','WL') for s in (0,1)]

def prepare(plan,out):
 out=Path(out)
 if out.exists():raise ValueError('Fresh preparation required')
 check(plan['hashes']);idx=np.load(plan['eval']['indices']);tc=plan['training']['0'];rows=Rows(tc['data'],tc['model']['length']);allids=np.arange(len(rows.ids));clusters=cluster_ids(rows,allids)
 forbidden=np.unique(clusters[np.r_[idx['fit'],idx['valid']]])
 if np.intersect1d(clusters[idx['fit']],clusters[idx['valid']]).size:raise ValueError('Router document overlap')
 eligible=rows.eligible();pool=eligible[~np.isin(clusters[eligible],forbidden)];out.mkdir(parents=True);orderhashes={}
 for seed in (0,1):
  t=plan['training'][str(seed)];needed=t['steps']*t['batch']
  if needed>len(pool):raise ValueError('Insufficient disjoint training rows')
  selected=np.random.default_rng(1901+seed).choice(pool,needed,replace=False);p=out/f'train-s{seed}.npy';np.save(p,selected);orderhashes[str(p)]=sha(p)
  assert np.intersect1d(clusters[selected],forbidden).size==0
 for name in NAMES:
  arm,s=name.split('-');seed=int(s[1:]);t=dict(plan['training'][str(seed)],arm=arm,seed=seed,indices=str(out/f'train-s{seed}.npy'));path_for(arm,0,t['steps'])
  t['expected_inputs']={p:sha(p) for p in (t['data']+'.bin',t['data']+'.starts',t['indices'])}
  write_json(out/(name+'.json'),dict(name=name,train=t,eval=plan['eval'],hashes={**plan['hashes'],**orderhashes},router_epochs=plan.get('router_epochs',30)))
 write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],training_router_document_overlap=0,eligible_rows=len(pool),files={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print('PROGRESSIVE_INPUTS_READY',sha(out/'complete.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
