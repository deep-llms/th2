import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);dest=Path(sys.argv[2]);result={'scope':'Reused DCLM rows; exploratory, two seeds; per-input adapted donor required for most patches','seeds':{}}
for seed in ('0','1'):
 study=root/'prepared'/f's{seed}'/'study'
 report=json.loads((study/'REPORT.json').read_text());cal=json.loads((study/'calibration.json').read_text());panels={}
 with np.load(study/'scores.npz',allow_pickle=False) as raw:
  den=raw['token_count'].sum()
  for b in ('A','B'):
   panels[b]={'calibration':{k:cal['panels'][b][k] for k in ('actual_rank','observations','centered_energy_fraction','mean_squared_norm','boundary_eigengap')},'gains':{c.split('/',1)[1]:v for c,v in report['gains'].items() if c.startswith(b+'/')},'paired_comparisons':{}}
   pairs=[('projected','complement'),('affine','centered'),('projected','radial'),('norm_restored','base'),('full','projected')]+[('projected',f'random_energy/{s}') for s in (20261121,20261122,20261123)]
   for x,y in pairs:
    d=raw[b+'/'+y]-raw[b+'/'+x];row=d/raw['token_count']
    panels[b]['paired_comparisons'][x+'_over_'+y]={'gain_nat':float(d.sum()/den),'q10_median_q90':np.quantile(row,[.1,.5,.9]).tolist(),'positive_rows':int((d>0).sum()),'negative_rows':int((d<0).sum())}
 result['seeds'][seed]=panels
print(json.dumps({seed:{b:{'centered_energy_fraction':v['calibration']['centered_energy_fraction'],'gains':{k:d['gain_nat'] for k,d in v['gains'].items()}} for b,v in panels.items()} for seed,panels in result['seeds'].items()},indent=2))
dest.write_text(json.dumps(result,indent=2)+'\n')
