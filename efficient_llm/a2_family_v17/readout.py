"""Append owner-requested results without rewriting registered F0–F6 readings."""
from pathlib import Path
import csv
import json
import numpy as np
from .common import read,write,SPEC_SHA
from .protocol import runs,capacity
from efficient_llm.a2_family_f6.readout import adapted
from efficient_llm.a2_family_f6.protocol import reading
from scripts.a2_family_report import pair_scores,arrays
from scripts.a2_family_v14_gate import read_lock,sha
from scripts.a2_quant_io import write as native_write


def analyze(results,history,repeats=10000):
    good={};rows=[];bad=[]
    for r in results:
        try:
            item=r['registered_run']
            if item not in runs() or r['spec_sha256']!=SPEC_SHA:raise ValueError('Unregistered result')
            v=adapted(r);par,c=arrays(v['baseline']['final']);pre,_=arrays(v['pre']['final']);post,_=arrays(v['scores']['final'])
            avg=lambda x:float(x@c/c.sum())
            rows.append(dict(**item,parent=avg(par),NLL=avg(post),delta_pre=avg(pre-par),delta_post=avg(post-par),flags=r.get('flags',[]),node='wc69',spec_sha256=SPEC_SHA))
            good[item['run_id']]=(r,v)
        except Exception as e:bad.append(dict(run_id=r.get('run_id'),error=repr(e)))
    missing=[r['run_id'] for r in runs() if r['run_id'] not in good]
    pairs={}
    for phase,cut,arms in [('F8',3,('P300','P1T')),('F8',9,('P300','P1T')),('F7',6,('S72','S188'))]:
        ids=[r['run_id'] for r in runs() if r['phase']==phase and len(r['removed'])==cut]
        if not all(i in good for i in ids):continue
        try:
            a,b=[good[i] for i in ids];v=pair_scores(a[1],b[1],repeats=repeats)
            if 'bootstrap' in v:v['bootstrap']['unit']='paired fixed-length packed sequences; native Why/F6 estimator'
            v.update(node='wc69',spec_sha256=SPEC_SHA,flags=[*a[0].get('flags',[]),*b[0].get('flags',[])])
            v['unverified']=bool(v['flags']);v['parent_gain_kept']=(v['G']-v['E'])/v['G'] if v['G'] else None
            delta=a[0]['delta_post'];v['E_over_short_delta']=v['E']/delta if delta else None
            pairs[f'{phase}-L{cut}']=v
        except Exception as e:bad.append(dict(pair=ids,error=repr(e)))
    table={int(k):dict(v) for k,v in history['cuts32'].items()}
    changes={}
    for cut in (3,9):
        if f'F8-L{cut}' in pairs:
            v=pairs[f'F8-L{cut}'];table[cut]=v
            difference=v['E']-history['E08'][str(cut)]
            changes[str(cut)]=dict(E08=history['E08'][str(cut)],E32=v['E'],difference=difference,
                label='flat' if abs(difference)<.003 else 'changing',direction='down' if difference<0 else 'up' if difference>0 else 'none')
    es={k:v['E'] for k,v in table.items()};complete=set(es)=={3,6,9,12}
    f8=dict(spec_sha256=SPEC_SHA,pairs={k:v for k,v in pairs.items() if k.startswith('F8')},four_cut_table=table,changes=changes,
            FY3_32B=capacity(es),tau=.003,secondary=True,original_Why_08_reading_unchanged='A not supported',
            slope_per_layer=float(np.polyfit([3,6,9,12],[es[k] for k in (3,6,9,12)],1)[0]) if complete else None,
            E12_minus_E3=es[12]-es[3] if complete else None,
            unverified=not complete or any(v.get('unverified',False) for v in table.values()),
            caveat='DCLM only; within-node E; historical L6 from 6y3o, other cuts wc69. Evaluation noise only, not document independence.')
    f7=dict(spec_sha256=SPEC_SHA,secondary=True,series=history['stable_series'],original_FE6_and_joint_reading_unchanged=True)
    if 'F7-L6' in pairs:
        pair=pairs['F7-L6'];old=history['stable_series']['dclm']
        f7.update(pair=pair,FE6_DCLM_32B=reading(old['0.8B']['E'],pair['E']),change02_to32=pair['E']-old['0.2B']['E'],unverified=pair['unverified'])
        f7['series']={**history['stable_series'],'dclm':{**old,'3.2B':pair}}
    else:f7.update(unverified=True,reason='Missing paired F7 results')
    return dict(F8=f8,F7=f7,rows=rows,missing_or_invalid=bad,missing_runs=missing,coverage_complete=not missing and not bad)


def report(cfg,events,final=False):
    root=Path(cfg['results_root']);results=[read(p) for p in sorted(root.glob('heals/*/result.json'))]
    d=analyze(results,cfg['historical_v17'],10000 if final else 0);d.update(events=events)
    old=read_lock(cfg['previous_decision'])
    # Preserve every original top-level value, including the old provenance SHA.
    combined=dict(old,F8=d['F8'],F7=d['F7'],v17_addendum=dict(spec_sha256=SPEC_SHA,**d,previous=cfg['previous_decision'],pause=cfg['depth_pause'],FT10=cfg['FT10']))
    if any(combined[k]!=v for k,v in old.items()):raise ValueError('Historical values changed')
    native_write(root/'decision_family.json',combined)
    write(root/'decision_v17.json',d)
    if final:
        prior=Path(cfg['previous_report']['path'])
        if sha(prior)!=cfg['previous_report']['sha256']:raise ValueError('Prior report changed')
        text=prior.read_text()+'\n\n# Family v1.7 — F8 and F7\n\n'
        text+=f'Spec SHA256: `{SPEC_SHA}`. Owner-requested extension; original F0–F6 values/readings/flags stand unchanged. FT10 skipped: same wc69 node.\n\n'
        text+='| Run | Parent | NLL | Δpre | Δpost | Flags |\n|---|---:|---:|---:|---:|---:|\n'
        for r in d['rows']:text+=f"| {r['run_id']} | {r['parent']:.9f} | {r['NLL']:.9f} | {r['delta_pre']:+.9f} | {r['delta_post']:+.9f} | {len(r['flags'])} |\n"
        text+='\nPaired sequence bootstrap: 10,000 resamples, seed 0; packed rows have 4095 scored targets each. Evaluation noise only. Full addendum quantities, provenance, missing runs and preserved flags:\n\n```json\n'+json.dumps(dict(d,pause=read_lock(cfg['depth_pause'])),indent=2)+'\n```\n'
        (root/'REPORT.md').write_text(text)
        with (root/'F8_F7.csv').open('w') as f:
            keys=['run_id','node','parent','NLL','delta_pre','delta_post','steps','tokens','lr','seed','stream_seed','spec_sha256'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(d['rows'])
        checks=[]
        for p in root.glob('heals/*/tests-rank*.json'):checks.extend(dict(c,path=str(p)) for c in read(p)['checks'])
        write(root/'tests.json',dict(checks=checks,failed=[c for c in checks if not c['passed']]))
    return d
