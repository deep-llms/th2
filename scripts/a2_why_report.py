"""Readouts from saved results, preserving missing or flagged measurements."""
from pathlib import Path
import math
from efficient_llm.a2_why.common import read,write,SPEC_SHA
from efficient_llm.a2_why.protocol import DATASETS,noise,merge_reading,capacity_reading,extend_capacity,spearman,descriptor


def finite(v):return isinstance(v,(int,float)) and math.isfinite(v)


def pair(root,phase,ds,**kw):
    rows={}
    for arm in ('P300','P1T'):
        item=descriptor(phase,ds,arm,**kw);path=Path(root)/'heals'/item['run_id']/'result.json'
        rows[arm]=read(path) if path.exists() else None
    valid=all(r is not None and finite(r.get('delta_post')) for r in rows.values())
    e=rows['P1T']['delta_post']-rows['P300']['delta_post'] if valid else None
    denom=rows['P300']['delta_post'] if valid else None
    return dict(E=e,E_relative=e/denom if finite(denom) and denom else None,
        P300_delta=denom,unverified=not valid or any(r.get('unverified') for r in rows.values() if r),
        runs={k:None if r is None else {n:r.get(n) for n in ('run_id','delta_post','flags','roundtrip_difference')} for k,r in rows.items()})


def part1(cfg):
    root=Path(cfg['results_root']);ref=cfg['historical']['datasets'];rows={}
    for ds in DATASETS:
        rows[ds]={'2030':ref[ds]['0.2B'],**{str(seed):pair(root,'Y1',ds,seed=seed) for seed in (2031,2032)}}
    available=all(finite(v['E']) for d in rows.values() for v in d.values())
    result=noise({ds:[v['E'] for v in rows[ds].values()] for ds in DATASETS}) if available else dict(sigma=None,tau=None,datasets={},std_ddof=1)
    result.update(pairs=rows,coverage_complete=available,unverified=not available or any(v.get('unverified',False) for d in rows.values() for v in d.values()),clause='tau=max(2*mean(sample_std_dataset),0.003); all three stream E values, ddof=1 fixed before new fits')
    write(root/'Y1_noise.json',result)
    note=f"Y1: sigma={result['sigma']}; tau={result['tau']}; sample std ddof=1; coverage={available}; unverified={result['unverified']}."
    (root/'Y1_NOTE.md').write_text(note+'\n\nSpec SHA256: '+SPEC_SHA+'\n');print('WHY_NOTE',note,flush=True)
    return result


def part2(cfg,n,post_note=True):
    root=Path(cfg['results_root']);result={}
    for variant in ('rdsc','avg'):
        values={ds:pair(root,'Y2',ds,variant=variant) for ds in DATASETS}
        available=finite(n['tau']) and all(finite(v['E']) for v in values.values())
        reading=merge_reading({ds:v['E'] for ds,v in values.items()},{ds:n['datasets'][ds]['mean'] for ds in DATASETS},n['tau']) if available else 'unverified'
        result[variant]=dict(pairs=values,reading=reading,extension=reading=='B supported',
            unverified=not available or n['unverified'] or any(v['unverified'] for v in values.values()),
            clause='B supported iff Evariant <= mean(Y1 E)-tau on both; no effect iff absolute difference < tau on both; else mixed')
    write(root/'Y2_readings.json',dict(variants=result))
    note='Y2: '+ '; '.join(f'{v}: {r["reading"]}, unverified={r["unverified"]}' for v,r in result.items())
    if post_note:
        (root/'Y2_NOTE.md').write_text(note+'\n\nSpec SHA256: '+SPEC_SHA+'\n');print('WHY_NOTE',note,flush=True)
    return result


def part3(cfg,n):
    root=Path(cfg['results_root']);ref=cfg['historical']['datasets'];budgets={}
    for budget in ('0.2B','0.8B'):
        values={ds:{k:ref[ds][budget] if k==6 else pair(root,'Y3',ds,layers=k,budget=budget) for k in (3,6,9,12)} for ds in DATASETS}
        available=all(finite(r['E']) for d in values.values() for r in d.values())
        es={ds:{k:v['E'] for k,v in d.items()} for ds,d in values.items()}
        budgets[budget]=dict(pairs=values,reading=capacity_reading(es,n['tau']) if available and finite(n['tau']) else 'unverified',
            strict_order={ds:all(es[ds][a]<es[ds][b] for a,b in ((3,6),(6,9),(9,12))) if available else None for ds in DATASETS},
            spearman={ds:spearman([es[ds][k] for k in (3,6,9,12)]) if available else None for ds in DATASETS},
            unverified=not available or n['unverified'] or any(r.get('unverified',False) for d in values.values() for r in d.values()))
    ext={}
    for ds in DATASETS:
        a=budgets['0.2B']['pairs'][ds][12]['E'];b=budgets['0.8B']['pairs'][ds][12]['E']
        ext[ds]=dict(trigger=extend_capacity(a,b) if finite(a) and finite(b) else None,E_02=a,E_08=b,clause='abs(E08-E02)/abs(E02)>0.20; zero baseline triggers iff E08 nonzero')
    result=dict(budgets=budgets,extensions=ext,reading=budgets['0.8B']['reading'])
    write(root/'Y3_readings.json',result);return result


def final(cfg,events):
    root=Path(cfg['results_root']);n=part1(cfg);b=part2(cfg,n);a=part3(cfg,n)
    extensions={'Y2':{},'Y3':{}}
    for v,r in b.items():
        if r['extension']:extensions['Y2'][v]={ds:dict(pair=pair(root,'Y2',ds,variant=v,budget='0.8B'),E_drop=cfg['historical']['datasets'][ds]['0.8B']['E']) for ds in DATASETS}
    for ds,r in a['extensions'].items():
        if r['trigger']:extensions['Y3'][ds]=pair(root,'Y3',ds,layers=12,budget='3.2B')
    checks=[];rows=[]
    for p in sorted(root.glob('heals/*/tests-rank*.json')):checks.extend(dict(x,path=str(p)) for x in read(p)['checks'])
    for p in sorted(root.glob('heals/*/result.json')):
        r=read(p);rows.append(dict(run_id=r['run_id'],dataset=r['dataset'],arm=r['arm'],val=r['scores']['final']['loss'] if r['scores']['final'] else None,
            PPL=math.exp(r['scores']['final']['loss']) if r['scores']['final'] and r['scores']['final']['loss']<709 else None,delta_pre=r['pre']['final']['loss']-r['baseline']['loss'],delta_post=r['delta_post'],unverified=r['unverified']))
    complete=all(e.get('returncode')==0 for e in events) and len(rows)==len(events)
    decision=dict(Y1=n,Y2=b,Y3=a,extensions=extensions,coverage_complete=complete,
        flags=[x for x in checks if not x['passed']],runs=rows,events=events,limitations=['One training seed; stream seeds2030/2031/2032','Why precedes remaining Depth by explicit user authorization','GPU resume of paused Depth still requires actual verification'])
    write(root/'tests.json',dict(checks=checks));write(root/'decision_why.json',decision)
    lines=['# A2 Why','',f'Spec SHA256: `{SPEC_SHA}`','',f'Coverage complete: {complete}. sigma={n["sigma"]}; tau={n["tau"]}.',
        '',f'Capacity reading at 0.8B: {a["reading"]}.','',*[f'{v}: {r["reading"]}; unverified={r["unverified"]}.' for v,r in b.items()],
        '','| Run | Final NLL | Delta pre | Delta post | Unverified |','|---|---:|---:|---:|---|']
    lines += [f'| {r["run_id"]} | {r["val"]} | {r["delta_pre"]} | {r["delta_post"]} | {r["unverified"]} |' for r in rows]
    lines += ['', 'Complete decisions, references, extension clauses, flags and per-sequence arrays are in decision_why.json and perseq/.',
        'Remaining: resume saved P300 Depth, P1T, downstream, independent replay and final Depth report.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return decision
