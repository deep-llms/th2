"""Hash-locked existing checkpoints only; no retraining or parent-output mutation."""
import argparse,json
from pathlib import Path
from .window_order_prepare import NAMES
from .strength_refinement import check
from .train import write_json
from .data import sha

def prepare(plan,out):
 out=Path(out)
 if out.exists():raise ValueError('Fresh recovery preparation required')
 check(plan['hashes']);out.mkdir(parents=True)
 for name in NAMES:
  parent=Path(plan['parent'])/'runs'/name;c=json.loads((parent/'config.json').read_text());windowed=c['train']['arm']!='D6';weights=parent/'train/weights.pt' if windowed else Path(c['existing_weights']);needed=set(c['hashes'])|{str(weights)}|set(plan['lineage'].values())
  if windowed:needed.add(str(parent/'train/fullstate.pt'))
  c.update(parent=str(parent),weights=str(weights),lineage=plan['lineage'],hashes={p:plan['hashes'][p] for p in needed});write_json(out/(name+'.json'),c)
 write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print('WINDOW_RECOVERY_INPUTS_READY',sha(out/'complete.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
