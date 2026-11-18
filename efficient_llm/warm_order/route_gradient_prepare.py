"""Deterministic fresh training rows; fixed repeated final sets, no score selection."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import Rows,sha
from .evaluate import cluster_ids
from .strength_refinement import check
from .train import write_json
NAMES=[f'{arm}-s{s}' for arm in ('HG','SG','GF','D6C') for s in (0,1)]

def prepare(plan,out):
 out=Path(out)
 if out.exists():raise ValueError('Fresh preparation required')
 check(plan['hashes'])
 if any(json.loads(Path(plan['parent_report']).read_text())['criterion'].values()):raise ValueError('Candidate requires negative causal-cost screen')
 ev=plan['eval'];ix=np.load(ev['indices']);tc=plan['training'];rows=Rows(tc['data'],tc['model']['length']);excluded=[ix['fit'],ix['valid']]
 checked={}
 for manifest in plan['prior_input_manifests']:
  base=Path(manifest).parent;proof=json.loads(Path(manifest).read_text())
  for rel,h in proof['files'].items():
   p=base/rel
   if sha(p)!=h:raise ValueError('Prior reservation changed '+str(p))
   checked[str(p)]=h
   if p.suffix=='.npy' and p.name.startswith('train'):excluded.append(np.load(p))
   if p.suffix=='.npz':
    a=np.load(p)
    excluded.extend(a[k] for k in ('fit','valid') if k in a)
 excluded=np.unique(np.concatenate(excluded));pool=np.setdiff1d(rows.eligible(),excluded)
 # Avoid documents of router fit/validation; prior training documents may recur.
 held_clusters=cluster_ids(rows,np.r_[ix['fit'],ix['valid']]);pool=pool[~np.isin(cluster_ids(rows,pool),held_clusters)]
 n=tc['batch']*tc['steps'];selected=np.random.default_rng(71209).permutation(pool)[:n]
 if len(selected)!=n:raise ValueError('Insufficient fresh rows')
 out.mkdir(parents=True)
 write_json(out/'reservation.json',dict(seed=71209,selected_rows=n,excluded_rows=len(excluded),router_document_overlap=0,prior_files=checked,scope='Fresh rows; prior training documents may recur. All arms/seeds share the selected row set; seed changes order.'))
 for seed in (0,1):
  ids=out/f'train-s{seed}.npy';np.save(ids,np.random.default_rng(71210+seed).permutation(selected))
  for arm in ('HG','SG','GF','D6C'):
   c=dict(tc,arm=arm,seed=seed,window=plan['window'],stride=plan['stride'],indices=str(ids))
   parent='LF' if arm in ('HG','SG') else arm;c['initial_weights']=plan['weights'][parent][str(seed)]
   c['expected_inputs']={p:sha(p) for p in (c['data']+'.bin',c['data']+'.starts',str(ids),c['initial_weights'])}
   hashes={**plan['hashes'],str(ids):sha(ids)}
   write_json(out/f'{arm}-s{seed}.json',dict(name=f'{arm}-s{seed}',train=c,eval=ev,hashes=hashes))
 write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()},fresh_training_rows=n,router_document_overlap=0))
 print('ROUTE_GRADIENT_INPUTS_READY',sha(out/'complete.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
