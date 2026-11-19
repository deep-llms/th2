"""Fresh recorded continuation rows, excluding both parent row orders and router documents."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import Rows,sha
from .evaluate import cluster_ids
from .strength_refinement import check
from .train import write_json
NAMES=[f'{a}-s{s}' for a in ('D','U','G','A') for s in (0,1)]
def prepare(plan,out):
 out=Path(out)
 if out.exists():raise ValueError('Fresh preparation required')
 check(plan['hashes']);prev=Path(plan['previous_inputs']);pin=json.loads(prev.read_text())
 for rel,h in pin['files'].items():
  if sha(prev.parent/rel)!=h:raise ValueError('Parent reservation changed')
 idx=np.load(plan['eval']['indices']);tc=plan['training']['0'];rows=Rows(tc['data'],tc['model']['length']);clusters=cluster_ids(rows,np.arange(len(rows.ids)));forbidden=np.unique(clusters[np.r_[idx['fit'],idx['valid']]])
 if np.intersect1d(clusters[idx['fit']],clusters[idx['valid']]).size:raise ValueError('Router document overlap')
 parentrows=np.unique(np.concatenate([np.load(prev.parent/f'train-s{s}.npy') for s in (0,1)]));pool=np.setdiff1d(rows.eligible(),parentrows);pool=pool[~np.isin(clusters[pool],forbidden)]
 out.mkdir(parents=True);orderhashes={}
 for seed in (0,1):
  tc=plan['training'][str(seed)];needed=tc['steps']*tc['batch']
  if needed>len(pool):raise ValueError('Not enough fresh rows')
  chosen=np.random.default_rng(1931+seed).choice(pool,needed,replace=False);p=out/f'train-s{seed}.npy';np.save(p,chosen);orderhashes[str(p)]=sha(p)
 for name in NAMES:
  arm,s=name.split('-');seed=int(s[1:]);t=dict(plan['training'][str(seed)],arm=arm,seed=seed,indices=str(out/f'train-s{seed}.npy'),initial_weights=plan['weights'][str(seed)])
  t['expected_inputs']={p:sha(p) for p in (t['data']+'.bin',t['data']+'.starts',t['indices'],t['initial_weights'])}
  write_json(out/(name+'.json'),dict(name=name,train=t,eval=plan['eval'],hashes={**plan['hashes'],**orderhashes}))
 write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],parent_training_row_overlap=0,router_document_overlap=0,eligible_rows=len(pool),files={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print('AUXILIARY_INPUTS_READY',sha(out/'complete.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
