"""Describe the registered terminal correction endpoints, not a parameter search."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(root,dest):
    root,dest=Path(root),Path(dest);result=dict(scope='Reused exploratory DCLM, two parents, one block/swap; no generalization or universal portability',seeds={})
    rows=[]
    for seed in ('0','1'):
        study=root/'prepared'/f's{seed}'/'study';report=json.loads((study/'REPORT.json').read_text());manifest=json.loads((study/'manifest.json').read_text())
        cases=report['cases'];panel=dict(cases=cases,vector_norms=manifest['vector_norms'],fits={})
        for b in ('C','S'):
            fit=root/'prepared'/f's{seed}'/'work'/b/'fit'
            panel['fits'][b]=json.loads((fit/'complete.json').read_text())
        with np.load(study/'scores.npz',allow_pickle=False) as raw:
            count=raw['token_count'];delta=(raw['S/mu_B']-raw['S/learned_S'])/count
            panel['learned_S_over_fixed_mu_B']=dict(gain_nat=float(((raw['S/mu_B']-raw['S/learned_S']).sum())/count.sum()),
                positive_rows=int((delta>0).sum()),q10_median_q90=np.quantile(delta,[.1,.5,.9]).tolist())
        panel['remaining_swap_gap_vs_native_C']=cases['S/learned_S']['nll_nat']-cases['C/zero']['nll_nat']
        panel['remaining_swap_gap_vs_learned_C']=cases['S/learned_S']['nll_nat']-cases['C/learned_C']['nll_nat']
        result['seeds'][seed]=panel
        rows += [dict(seed=seed,case=k,**{key:v[key] for key in ('nll_nat','gain_vs_zero_nat','positive_rows','negative_rows')}) for k,v in cases.items()]
    dest.write_text(json.dumps(result,indent=2)+'\n')
    with dest.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['S/mu_B','S/learned_S','C/learned_C','C/learned_S','S/learned_C']
    names=['Fixed donor mean\non swapped path','Learned S\non swapped path','Learned C\non canonical path','Learned S\non canonical path','Learned C\non swapped path']
    fig,ax=plt.subplots(figsize=(10,4.4));x=np.arange(len(labels));width=.36
    for seed,color in [('0','#176DA8'),('1','#E79232')]:
        values=[result['seeds'][seed]['cases'][k]['gain_vs_zero_nat'] for k in labels]
        ax.bar(x+(int(seed)-.5)*width,values,width,label='Seed '+seed,color=color)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(x,names,fontsize=9)
    ax.set_ylabel('NLL reduction vs same-path zero bias (nat/token)')
    ax.set_title('Frozen-body constant correction: strong path-specific benefit, negative transfer')
    ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.01,.01,'Exploratory: reused 512 rows; body frozen, 768 learned parameters; 500 updates per fit. No portability/generalization claim.',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(dest.with_suffix('.png'),dpi=160);plt.close(fig)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('dest');a=p.parse_args();r=summarize(a.root,a.dest)
    print(json.dumps({s:dict(swap_gain=v['cases']['S/learned_S']['gain_vs_zero_nat'],canonical_cost=-v['cases']['C/learned_S']['gain_vs_zero_nat'],remaining_gap=v['remaining_swap_gap_vs_native_C']) for s,v in r['seeds'].items()}))
