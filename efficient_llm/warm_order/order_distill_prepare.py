"""Reuse hash-locked pilot rows and held-out router reservations without scores."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import Rows,sha
from .evaluate import cluster_ids
from .strength_refinement import check
from .train import write_json
NAMES=[f'{arm}-s{s}' for arm in ('C2','K2','P2','PK2') for s in (0,1)]

def prepare(plan,out):
 out=Path(out)
 if out.exists():raise ValueError('Fresh preparation required')
 check(plan['hashes']);idx=np.load(plan['eval']['indices']);trainids=[np.load(plan['training'][str(s)]['indices']) for s in (0,1)]
 if any(np.intersect1d(t,np.r_[idx['fit'],idx['valid']]).size for t in trainids):raise ValueError('Training/router overlap')
 rows=Rows(plan['eval']['stems']['fit'],plan['training']['0']['model']['length'])
 if np.intersect1d(cluster_ids(rows,idx['fit']),cluster_ids(rows,idx['valid'])).size:raise ValueError('Router document overlap')
 out.mkdir(parents=True)
 for name in NAMES:
  arm,seed=name.split('-');seed=int(seed[1:]);tc=dict(plan['training'][str(seed)]);tc['arm']=arm
  if tc['steps']%4 or tc['steps']<4:raise ValueError('Balanced depth cycle requires multiple of four')
  cfg=dict(name=name,train=tc,eval=plan['eval'],hashes=plan['hashes'],router_epochs=plan.get('router_epochs',30))
  write_json(out/(name+'.json'),cfg)
 write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],router_train_overlap=0,router_fit_valid_doc_overlap=0,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print('ORDER_DISTILL_INPUTS_READY',sha(out/'complete.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
