"""Frozen extra-block LR screen using the unchanged tested refinement trainer."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .data import Rows,sha
from .train import write_json
from .model import Config,Model
from .refinement import Refinement
from .refinement_train import train
from .refinement_eval import check_train,collect,probe
from .refinement_probe import numerical
from .evaluate import cluster_ids
from .token_router import precision
NAMES=[f'{arm}-s{s}' for arm in ('H3','H10') for s in (0,1)]
LRS={'H3':3e-4,'H10':1e-3}


def check(hashes):
 for p,h in hashes.items():
  if sha(p)!=h:raise ValueError('Changed input '+p)


def prepare(plan,out):
 out=Path(out)
 if out.exists():raise ValueError('Fresh inputs required')
 check(plan['hashes']);out.mkdir(parents=True)
 for name in NAMES:
  arm,seed=name.split('-');seed=int(seed[1:]);old=plan['parents'][str(seed)]
  recipe=json.loads(Path(old['recipe']).read_text());c=dict(recipe['config'])
  if c['arm']!='F7' or c['seed']!=seed:raise ValueError('Wrong frozen control')
  check(c['expected_inputs']);c['lr']=LRS[arm]
  cfg=dict(name=name,train=c,eval=json.loads(Path(old['eval_config']).read_text()),prior=old['prior'],c6=old['c6'],hashes=plan['hashes'])
  write_json(out/(name+'.json'),cfg)
 write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
 print('STRENGTH_REFINEMENT_INPUTS_READY',sha(out/'complete.json'),flush=True)


def run(c,out,device='cuda'):
 out=Path(out)
 if out.exists():raise ValueError('Fresh job required')
 if device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('One GPU required')
 check(c['hashes']);precision();out.mkdir(parents=True);write_json(out/'config.json',c)
 start=time.monotonic();proof=numerical(device);tc=c['train'];td=out/'train'
 train(tc,td,device,stop_after=min(40,tc['steps']-1))
 first=json.loads((td/'checkpoint.json').read_text());proof['first_checkpoint']=first
 # Exact same recipe, RNG, optimizer and fixed schedule; no weight-only restart.
 train(tc,td,device,resume=True);done,_=check_train(td)
 state=torch.load(td/'weights.pt',map_location='cpu',weights_only=False)
 initial=torch.load(tc['initial_weights'],map_location='cpu',weights_only=False)
 for k,v in initial['model'].items():
  if not torch.equal(v,state['model']['base.'+k]):raise ValueError('Frozen base mutated '+k)
 m=Refinement(Model(Config(**tc['model'])),frozen=True);m.load_state_dict(state['model']);m.to(device).eval()
 for p in m.parameters():p.requires_grad_(False)
 ev=c['eval'];check(ev['hashes']);ids=np.load(ev['indices']);rows={k:Rows(v,tc['model']['length']) for k,v in ev['stems'].items()}
 proof['production']=probe(m,rows['fit'].batch(ids['valid'][:2],device));proof['frozen_base_exact']=True
 for domain in ('valid','dclm','fwe'):
  selected=ids[domain];rr=rows['fit' if domain=='valid' else domain]
  values,_=collect(m,rr,selected,out/domain,device)
  arrays=dict(ids=selected,clusters=cluster_ids(rr,selected),base=values[:,:,0].mean(1,dtype=np.float64),refine=values[:,:,1].mean(1,dtype=np.float64),oracle=values.min(-1).mean(1,dtype=np.float64))
  old=Path(c['prior'])/domain
  if domain=='valid':
   prior=np.load(old/'costs.npy',mmap_mode='r');arrays['low_lr']=prior[:,:,1].mean(1,dtype=np.float64)
  else:
   oldarr=np.load(old/'summary.npz');ctrl=np.load(c['c6'][domain]);np.testing.assert_array_equal(selected,oldarr['ids']);np.testing.assert_array_equal(selected,ctrl['ids'])
   np.testing.assert_allclose(arrays['base'],oldarr['base'],rtol=0,atol=2e-4)
   arrays.update(low_lr=oldarr['refine'],c6=ctrl['dense'])
  np.savez(out/domain/'summary.npz',**arrays)
 check(c['hashes']);check_train(td)
 write_json(out/'complete.json',dict(status='complete',name=c['name'],seconds=time.monotonic()-start,train=done,proof=proof,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.pt'}))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--config');p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
 if bool(a.plan)==bool(a.config):raise ValueError('Exactly one operation')
 if a.plan:prepare(json.loads(Path(a.plan).read_text()),a.out)
 else:run(json.loads(Path(a.config).read_text()),a.out,a.device)
