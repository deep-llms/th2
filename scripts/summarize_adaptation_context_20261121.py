"""Summarize verified context scores; no model execution or parameter fitting."""
import json
from pathlib import Path
import numpy as np
root=Path('results/layer_dependency_20261120/adaptation_context_export')
out={}
for seed in (0,1):
 p=root/'prepared'/f's{seed}'/'study'
 r=json.loads((p/'REPORT.json').read_text());out[str(seed)]=r
 print('SEED',seed,'cases',json.dumps(r['cases']))
 for k in r['effects']['B']:
  a=r['effects']['A'][k];b=r['effects']['B'][k]
  print(k,'A',round(a['gain_nat'],6),'B',round(b['gain_nat'],6),'B positive',b['positive_rows'],'q',b['q10_median_q90'])
 # Matched difference of measured B and A interventions, descriptive only.
 with np.load(p/'scores.npz') as x:
  den=x['token_count']; contrasts={}
  for name,lhs,rhs in [('input_at_parent','000','100'),('block_at_parent','000','010'),('reader_at_parent','000','001'),('input_at_adapted','011','111'),('block_at_adapted','101','111'),('reader_at_adapted','110','111')]:
   v=(x['B/'+lhs]-x['B/'+rhs])-(x['A/'+lhs]-x['A/'+rhs])
   contrasts[name]=dict(gain_nat=float(v.sum()/den.sum()),q10_median_q90=np.quantile(v/den,[.1,.5,.9]).tolist(),positive_rows=int((v>0).sum()))
  out[str(seed)]['matched_B_minus_A']=contrasts
Path('results/layer_dependency_20261120/adaptation_context_summary.json').write_text(json.dumps(out,indent=2)+'\n')
