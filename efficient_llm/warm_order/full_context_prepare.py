"""Reuse recorded common rows and verified dense endpoints, without final-score selection."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import sha
from .strength_refinement import check
from .train import write_json
NAMES=[f'{a}-s{s}' for a in ('F','H','S') for s in (0,1)]
ALL_NAMES=NAMES+[f'D6C-s{s}' for s in (0,1)]

def prepare(plan,out):
 out=Path(out)
 if out.exists():raise ValueError('Fresh preparation required')
 check(plan['hashes']);parent=json.loads(Path(plan['parent_report']).read_text())
 if parent['mechanism_all_seeds_domains'] or parent['continuation_all_seeds_domains']:raise ValueError('Resolve positive predecessor before changing architecture')
 previous=Path(plan['previous_inputs']);proof=json.loads(previous.read_text())
 for rel,h in proof['files'].items():
  if sha(previous.parent/rel)!=h:raise ValueError('Prior reservation changed')
 out.mkdir(parents=True)
 for name in NAMES:
  arm,s=name.split('-');seed=int(s[1:]);tc=dict(plan['training'][str(seed)],arm=arm,seed=seed,window=plan['window'],stride=plan['stride'],initial_weights=plan['weights'][str(seed)])
  if tc['steps']<2:raise ValueError('Resume probe needs two updates')
  ids=Path(tc['indices'])
  if ids.parent!=previous.parent or proof['files'].get(ids.name)!=sha(ids):raise ValueError('Unverified row order')
  tc['expected_inputs']={p:sha(p) for p in (tc['data']+'.bin',tc['data']+'.starts',str(ids),tc['initial_weights'])}
  hashes={**plan['hashes'],str(ids):sha(ids)}
  write_json(out/(name+'.json'),dict(name=name,train=tc,eval=plan['eval'],hashes=hashes))
 write_json(out/'reused.json',plan['reused'])
 write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()},row_scope='same recorded row sets/order as predecessor; pre-study dense parents have not consumed those rows',previous_inputs_sha256=sha(previous)))
 print('FULL_CONTEXT_INPUTS_READY',sha(out/'complete.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
