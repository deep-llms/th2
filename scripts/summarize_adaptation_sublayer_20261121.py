"""Recompute descriptive B-minus-A contrasts from verified paired score arrays."""
import json
from pathlib import Path
import numpy as np
from efficient_llm.warm_order.adaptation_sublayer_pipeline import replay

root=Path('results/layer_dependency_20261120/adaptation_sublayer_export')
out={}
for seed in (0,1):
    p=root/'prepared'/f's{seed}'/'study';replay(p)
    report=json.loads((p/'REPORT.json').read_text());out[str(seed)]=report
    with np.load(p/'scores.npz',allow_pickle=False) as raw:
        den=raw['token_count'];effects={}
        for branch in ('A','B'):
            x={k:raw[branch+'/'+k] for k in ('00','01','10','11','S')}
            effects[branch]={'attention':x['00']-x['10'],'ffn':x['00']-x['01'],
                             'joint':x['00']-x['11'],'interaction':x['10']+x['01']-x['00']-x['11']}
        contrasts={}
        for name in effects['B']:
            v=effects['B'][name]-effects['A'][name]
            contrasts[name]=dict(gain_nat=float(v.sum()/den.sum()),
                q10_median_q90=np.quantile(v/den,[.1,.5,.9]).tolist(),positive_rows=int((v>0).sum()),
                interpretation='Matched descriptive B-minus-A; no independent-document CI or causal share')
        out[str(seed)]['matched_B_minus_A']=contrasts
    print('SEED',seed,'cases',json.dumps(report['cases']))
    for k,b in report['effects']['B'].items():
        print(k,'B',b['gain_nat'],'A',report['effects']['A'][k]['gain_nat'],
              'positive_rows',b['positive_rows'],'q10_median_q90',b['q10_median_q90'])
Path('results/layer_dependency_20261120/adaptation_sublayer_summary.json').write_text(json.dumps(out,indent=2)+'\n')
