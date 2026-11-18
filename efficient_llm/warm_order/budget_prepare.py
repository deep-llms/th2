"""CPU hash admission for frozen cached counterfactuals; no new sampling or scores."""
import argparse,json
from pathlib import Path
from .data import sha
from .train import write_json
from .budget_eval import check_hashes

def prepare(plan,out):
 out=Path(out)
 if out.exists():raise ValueError('Fresh preparation required')
 hashes={**plan['common_hashes']}
 for job in plan['jobs']:hashes.update(job['hashes'])
 check_hashes(hashes);out.mkdir(parents=True)
 for job in plan['jobs']:write_json(out/(job['name']+'.json'),{**job,'hashes':{**plan['common_hashes'],**job['hashes']}})
 write_json(out/'complete.json',dict(status='ready',hashes=hashes,files={p.name:sha(p) for p in out.iterdir() if p.is_file()},jobs=len(plan['jobs']),scope='Frozen bodies; four gain heads and fixed validation budgets; no body retraining'))
 print('BUDGET_REFINEMENT_INPUTS_READY',sha(out/'complete.json'),(out/'complete.json').read_text(),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(json.loads(Path(a.plan).read_text()),a.out)
