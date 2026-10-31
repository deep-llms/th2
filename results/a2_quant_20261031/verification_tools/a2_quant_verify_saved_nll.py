"""Read-only local receipt audit; writes a new local verification file."""
import argparse,json,hashlib,math
from pathlib import Path
import numpy as np
from efficient_llm.a2.quant_stats import nll_report,PARENTS
p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
base=a.receipt/'results/a2_quant_20261031';cfg=json.loads((base/'config/plan.json').read_text())
expected={(ds,parent,comp,setname) for ds in ('dclm','fwe','nem') for parent in PARENTS
 for comp in ('parent','pre-S1','rtn4','rtn3','gptq4','gptq3')+ (('gptq3-cal1',) if ds!='nem' else ())
 for setname in ('final','own_corpus')}
assert len(expected)==160
found={};values={};records=[]
for pattern in ('workers/*/*/*/*-final.json','workers/*/*/*/*-own_corpus.json'):
 for path in sorted(base.glob(pattern)):
  row=json.loads(path.read_text());key=(row['dataset'],row['parent'],row['compression'],row['set'])
  assert key in expected and key not in found
  xs=np.asarray(row['per_sequence_nll'],dtype=np.float64);assert np.isfinite(xs).all()
  ds=cfg['sets'][row['dataset']][row['set']];assert len(xs)==ds['sequences']
  assert row['data_manifest']==ds['manifest'] and row['spec_sha256']==cfg['spec_sha256']
  assert row['code_commit']==cfg['code_commit'] and float(xs.mean())==row['loss']
  assert abs(math.fsum(xs)/len(xs)-row['loss'])<1e-12
  if row['compression'].startswith('gptq') and not cfg['calibration_passed']:assert row['unverified']
  found[key]=row['loss'];records.append(dict(key=key,nll=row['loss'],sequences=len(xs),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),unverified=row['unverified']))
  if row['set']=='final':values.setdefault(row['dataset'],{}).setdefault(row['compression'],{})[row['parent']]=xs.tolist()
decision_path=base/'decision_quant.json';ci_verified=None
if decision_path.exists():
 decision=json.loads(decision_path.read_text());actual=nll_report(values)
 ci_verified=actual['datasets']==decision['datasets'] and actual['across_datasets']==decision['across_datasets']
 assert ci_verified,'Saved per-sequence bootstrap differs from decision'
out=dict(spec_sha256=cfg['spec_sha256'],code_commit=cfg['code_commit'],expected=160,actual=len(found),coverage_complete=set(found)==expected,
 missing=sorted(expected-set(found)),independent_mean_checks=True,paired_CI_recomputed_exact=ci_verified,records=records)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'}))
