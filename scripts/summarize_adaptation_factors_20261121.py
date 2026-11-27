"""Replay and expose prespecified factor contrasts; no fitting or model execution."""
import json
from pathlib import Path
import numpy as np
from efficient_llm.warm_order.adaptation_factors_pipeline import replay
root=Path('results/layer_dependency_20261120/adaptation_factors_export');out={}
for seed in (0,1):
    p=root/'prepared'/f's{seed}'/'study';replay(p)
    r=json.loads((p/'REPORT.json').read_text());out[str(seed)]=r
    print('SEED',seed)
    for k,b in r['effects']['B'].items():
        print(k,'B',b['gain_nat'],'A',r['effects']['A'][k]['gain_nat'],
              'positive',b['positive_rows'],'q10_median_q90',b['q10_median_q90'])
    with np.load(p/'scores.npz',allow_pickle=False) as x:
        den=x['token_count'];extra={}
        for branch in ('A','B'):
            extra[branch]={}
            for family,zero,projection in [('attn','000','011'),('ffn','0000','0111')]:
                v=x[f'{branch}/{family}/{zero}']-x[f'{branch}/{family}/{projection}']
                extra[branch][family+'/all_projection_parent_norm']=dict(gain_nat=float(v.sum()/den.sum()),
                    q10_median_q90=np.quantile(v/den,[.1,.5,.9]).tolist(),positive_rows=int((v>0).sum()))
        out[str(seed)]['projection_only']=extra
        print('PROJECTION_ONLY',json.dumps(extra))
Path('results/layer_dependency_20261120/adaptation_factors_summary.json').write_text(json.dumps(out,indent=2)+'\n')
