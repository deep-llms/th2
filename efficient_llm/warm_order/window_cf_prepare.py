"""Lock frozen endpoints and validation-only intervention jobs."""
import argparse,json
from pathlib import Path
from .strength_refinement import check
from .train import write_json
from .data import sha
NAMES=[f'{a}-s{s}-{mode}' for a in ('GF','LF') for s in (0,1) for mode in ('hold','rescore')]
def prepare(plan,out):
 out=Path(out)
 if out.exists():raise ValueError('Fresh preparation required')
 check(plan['hashes'])
 if json.loads(Path(plan['parent_report']).read_text())['continuation_all_seeds_domains'] is not False:raise ValueError('Only after negative parent screen')
 out.mkdir(parents=True)
 for name in NAMES:
  arm,seed,mode=name.split('-');seed=int(seed[1:]);c=dict(name=name,arm='WG' if arm=='GF' else 'WL',seed=seed,mode=mode,sequences=plan['sequences'],data=plan['data'],indices=plan['indices'],weights=plan['weights'][arm][str(seed)],dense_weights=plan['weights']['D6C'][str(seed)],hashes=plan['hashes']);write_json(out/(name+'.json'),c)
 write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print('WINDOW_CF_INPUTS_READY',sha(out/'complete.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
