import json,hashlib,math,collections
from pathlib import Path
import numpy as np
from efficient_llm.a2.quant_stats import nll_report
root=Path('results/a2_mech_20261031/artifacts/node');base=root/'results/a2_mech_20261031';rev=base/'readout_frequency_revision'
manifest=json.loads(Path('results/a2_mech_20261031/artifacts/manifest.json').read_text())
load=lambda p:json.loads(p.read_text())
for rel,info in manifest['files'].items():
 raw=(root/rel).read_bytes();assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256'],rel
workers=list((base/'workers').glob('*/*/*/result.json'));counts=collections.Counter(p.parts[-4] for p in workers);assert counts==dict(weights=12,layers=12,noise=12,tokens=8)
assert all(load(p)['execution_complete'] for p in workers)
assert all(c['passed'] for p in workers for c in load(p)['checks'])
means_checked=0;max_mean_error=0
for p in (base/'workers').glob('*/*/*/*.json'):
 v=load(p)
 if not isinstance(v,dict) or 'per_sequence_nll' not in v:continue
 a=np.array(v['per_sequence_nll'],dtype=float);assert np.isfinite(a).all()
 expected=256 if '/layers/' in str(p) else 2461;assert a.shape==(expected,)
 error=abs(float(a.mean())-v['loss']);max_mean_error=max(error,max_mean_error);assert error<1e-12,p
 means_checked+=1
weights=load(rev/'weights.json')['parents'];assert len(weights)==12
for v in weights.values():
 assert len(v['matrices'])==168
 for k,mean in v['means'].items():
  calc=sum(r[k]*r['weights'] for r in v['matrices'])/sum(r['weights'] for r in v['matrices']);assert calc==mean
layers=load(rev/'layers.json')['parents'];assert len(layers)==12
for key,v in layers.items():
 folder=base/'workers/layers'/key;b=load(folder/'parent-small.json')['loss'];singles=[load(folder/f'single-{i}.json')['loss']-b for i in range(24)]
 assert singles==v['single_damage'];s=sum(singles[16:22]);assert s==v['S1_sum'];assert (load(folder/'pre-S1-small.json')['loss']-b)/s==v['rho']
d=load(rev/'decision_mech.json');metric_checks=0
for m,v in d['metrics'].items():
 for ds,t in v['relative_contrasts'].items():
  x=v['values'][ds]
  for name,a,b in [('L_ann','P1T','P300'),('L_stab','S188','S72'),('R_short','P300','S72')]:
   assert abs((x[a]-x[b])-t[name]['contrast'])<1e-12
   assert abs((x[a]-x[b])/x[b]-t[name]['relative'])<1e-12;metric_checks+=1
values={}
for ds in ('dclm','fwe','nem'):
 table={'parent':{}};values[ds]=table
 for par in ('S72','P300','S188','P1T'):
  folder=base/'workers/noise'/ds/par;table['parent'][par]=load(folder/'parent-final.json')['per_sequence_nll']
  for kind in ('N-iso-0.1','N-iso-0.2','N-iso-0.3','N-rtn3'):
   a=[load(folder/f'{kind}-seed{s}.json')['per_sequence_nll'] for s in (0,1)]
   table.setdefault(kind,{})[par]=np.mean(a,axis=0).tolist()
actual=nll_report(values);expected=load(rev/'noise_paired.json');expected.pop('spec_sha256')
errors=[]
def compare(x,y):
 if isinstance(x,dict):
  assert set(x)==set(y)
  for k in x:compare(x[k],y[k])
 elif isinstance(x,list):
  assert len(x)==len(y)
  for a,b in zip(x,y):compare(a,b)
 elif isinstance(x,float):errors.append(abs(x-y));assert math.isclose(x,y,rel_tol=1e-11,abs_tol=1e-12),(x,y)
 else:assert x==y,(x,y)
compare(actual,expected)
buckets=load(rev/'buckets.json')
for ds,table in buckets['datasets'].items():
 for kind,bs in table['buckets'].items():
  assert len(bs)==10 and sum(b['count'] for b in bs)==2461*4095
  for b in bs:
   m=b['means'];g=m['P300']-m['P1T'];s=m['student-P300']-m['student-P1T']
   assert g==b['parent_gap'] and s==b['student_gap'] and s/g==b['retention'];assert b['low_denominator']==(abs(g)<.005)
  r=[b['retention'] for b in bs];h=d['H_E_readout'][ds][kind]
  assert float(np.mean(r[:3]))==h['bottom3'] and float(np.mean(r[-3:]))==h['top3']
ver=load(rev/'verification.json');assert ver['coverage_complete'] and ver['independent_readout_exact']
tests=load(rev/'tests.json')['tests'];fail=[r for r in tests if not r['passed']];assert len(fail)==1 and fail[0]['test']=='original_coordinator_exit'
external=[r for r in tests if r['test'].startswith('MT6_external_replay:')];assert len(external)==18 and all(r['passed'] for r in external)
result=dict(spec_sha256=d['spec_sha256'],files_sha_verified=len(manifest['files']),worker_counts=counts,per_sequence_means_checked=means_checked,per_sequence_max_error=max_mean_error,matrix_rows_verified=2016,relative_contrasts_verified=metric_checks,paired_noise_bootstrap_resamples=10000,paired_noise_bootstrap_seed=0,paired_noise_bootstrap_max_error=max(errors),tests=len(tests),test_failures=fail,external_MT6_replays=18,canonical_readout_exact=True,token_arrays_on_node=len(manifest['token_arrays_retained_on_node']),token_arrays_downloaded=False,hypotheses=d['hypotheses'])
Path('results/a2_mech_20261031/final/local_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
