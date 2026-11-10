import json,argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
r={arm:json.loads((a.root/'heals'/f'ST1-dclm-{arm}-S1-12.8B-lr1e-3/result.json').read_text()) for arm in ['P300','P1T','X72']}
def arr(arm,parent=False):
 s=r[arm]['baseline'] if parent else r[arm]['scores']['final'];v=np.asarray(s['per_sequence_nll']);assert len(v)==2461 and s['tokens']==4095*len(v);assert np.isfinite(v).all();return v
p,t,x=arr('P300'),arr('P1T'),arr('X72');pp,tp=arr('P300',True),arr('P1T',True);e=(t-tp)-(p-pp);init=x-p
rng=np.random.default_rng(0);draw=[]
for _ in range(10000):
 ix=rng.integers(0,len(e),len(e));draw.append([float(e[ix].mean()),float(init[ix].mean())])
ci=np.quantile(draw,[.025,.975],axis=0);d=json.loads((a.root/'decision_depth_partc.json').read_text());assert abs(float(e.mean())-d['E'])<1e-12 and abs(float(init.mean())-d['A_init'])<1e-12
out=dict(E=dict(estimate=float(e.mean()),ci95=ci[:,0].tolist()),A_init=dict(estimate=float(init.mean()),ci95=ci[:,1].tolist()),resamples=10000,seed=0,sequences=2461,prediction_tokens=2461*4095,method='Independent paired packed-sequence bootstrap. Equal 4095 prediction tokens per row, so means are token-weighted. Evaluation uncertainty only; no independent-doc or training-seed claim.',spec_sha256=d['spec_sha256']);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
