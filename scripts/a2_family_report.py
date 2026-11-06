"""Family readout from persisted document arrays; no model execution."""
from pathlib import Path
import math
import numpy as np
from efficient_llm.a2_family.common import read,write,sha,SPEC_SHA
from efficient_llm.a2_family.protocol import break_even,changed,mixture_reading


def arrays(score):
    nll=np.asarray(score['per_sequence_nll'],dtype=np.float64)
    counts=np.asarray(score['per_sequence_tokens'],dtype=np.int64)
    if nll.ndim!=1 or nll.shape!=counts.shape or not np.isfinite(nll).all() or (counts<0).any() or counts.sum()<=0:
        raise ValueError('Invalid document arrays')
    if abs(float(nll@counts/counts.sum())-score['loss'])>1e-10:raise ValueError('Saved scalar/array mismatch')
    return nll,counts


def pair_scores(short,long,drop=(),repeats=10000):
    values=[];counts=None
    for r in (short,long):
        row=[]
        for score in (r['baseline']['final'],r['pre']['final'],r['scores']['final']):
            v,c=arrays(score)
            if counts is not None and not np.array_equal(c,counts):raise ValueError('Pair document token counts differ')
            counts=c;row.append(v)
        values.append(row)
    keep=np.ones(len(counts),dtype=bool);keep[list(drop)]=False
    if not keep.any() or counts[keep].sum()==0:raise ValueError('No kept prediction tokens')
    a,b=[np.asarray(v)[:,keep] for v in values];c=counts[keep]
    e=(b[2]-b[0])-(a[2]-a[0]);g=a[0]-b[0]
    ep=(b[1]-b[0])-(a[1]-a[0]);weighted=lambda x:float(x@c/c.sum())
    result=dict(E=weighted(e),E_pre=weighted(ep),G=weighted(g),dropped_sequences=int((~keep).sum()),dropped_prediction_tokens=int(counts[~keep].sum()),kept_sequences=int(keep.sum()),kept_prediction_tokens=int(c.sum()))
    result['E_over_G']=result['E']/result['G'] if abs(result['G'])>1e-15 else None
    draws=[];ratios=[];undefined=0;rng=np.random.default_rng(0)
    for _ in range(repeats):
        ix=rng.integers(0,len(c),len(c));den=float(c[ix].sum())
        if den==0:undefined+=1;continue
        ev=float(e[ix]@c[ix]/den);gv=float(g[ix]@c[ix]/den);draws.append(ev)
        if abs(gv)>1e-15:ratios.append(ev/gv)
    if repeats:result['bootstrap']=dict(E_ci95=np.quantile(draws,[.025,.975]).tolist() if draws else None,E_over_G_ci95=np.quantile(ratios,[.025,.975]).tolist() if ratios else None,resamples=repeats,seed=0,unit='paired documents',estimator='token weighted',zero_token_draws=undefined,undefined_ratio_draws=len(draws)-len(ratios))
    return result


def exposure(cfg,result):
    item=result['registered_run'];data=item['data'];seed=str(item['stream_seed'])
    audit=cfg.get('overlap',{}).get(data)
    if audit is None:return dict(status='unverified',reason='Whole-pool overlap audit unavailable')
    seen=set();count=0
    for record in result['raw_order_files']:
        if sha(record['path'])!=record['sha256']:raise ValueError('Actual sample-order trace changed')
        with Path(record['path']).open() as f:
            for line in f:
                import json
                ids=json.loads(line)['sample_ids'];count+=len(ids);seen.update(ids)
    expected=set(range(item['steps']*64))
    if seen!=expected or count!=len(expected):raise ValueError('Actual consumed row coverage differs from registered once-only prefix')
    records=audit['consumed_overlaps'][seed];observed=[]
    for doc in records:
        if any(row in seen for row in range(doc['stream_start']//4096,(doc['stream_stop']-1)//4096+1)):observed.append(doc)
    return dict(status='seen' if observed else 'not seen',documents=observed,actual_rows=len(seen),actual_tokens=len(seen)*4096)


def f0_readout(cfg,root):
    rows={};refs=cfg.get('historical',{}).get('datasets',{})
    for ds in ('dclm','fwe'):
        p=root/'reference'/ds/'result.json'
        if not p.exists():rows[ds]=dict(status='missing');continue
        r=read(p);score=r['scores'].get('final')
        if not score:rows[ds]=dict(status='missing_final_array');continue
        value=score['loss'];students=[];parent=None
        for budget,row in refs.get(ds,{}).items():
            for arm in ('P300','P1T'):
                if arm not in row.get('runs',{}):continue
                v=row['runs'][arm];parent=v['val_parent'] if arm=='P300' else parent
                students.append(dict(arm=arm,budget=budget,nll=v['val_post'],student_minus_1_3B=v['val_post']-value,total_tokens=(300e9 if arm=='P300' else 1e12)+float(budget[:-1])*1e9,provenance=v.get('provenance','historical Stage1 reference')))
        p300=[v for v in students if v['arm']=='P300']
        rows[ds]=dict(status='available',nll=value,parent_P300=parent,reference_minus_parent=None if parent is None else value-parent,students=students,break_even=break_even([float(v['budget'][:-1]) for v in p300],[v['nll'] for v in p300],value) if p300 else None,nonembedding_parameters=r['nonembedding_parameters'],reference_pretraining_tokens=300e9,flags=r['flags'],cross_node_12_8B_note='First-node final scalar; cross-node difference about 1e-4')
    return rows


def report(cfg,events,final=False):
    root=Path(cfg['results_root']);runs=[];bad=[];rows=[];groups={};tests=[]
    for p in sorted(root.glob('heals/*/result.json')):
        r=read(p);item=r['registered_run'];runs.append(r)
        try:
            par,c=arrays(r['baseline']['final']);pre,cp=arrays(r['pre']['final']);post,cs=arrays(r['scores']['final'])
            if not np.array_equal(c,cp) or not np.array_equal(c,cs):raise ValueError('Run document counts differ')
            exp=exposure(cfg,r)
            drop=cfg.get('overlap',{}).get(item['data'],{}).get('final_indices',[])
            keep=np.ones(len(c),dtype=bool);keep[drop]=False
            reduce=lambda v:float(v[keep]@c[keep]/c[keep].sum())
            rows.append(dict(run_id=r['run_id'],**{k:item[k] for k in ('arm','data','budget','lr','steps','tokens','stream_seed')},parent=float(par@c/c.sum()),NLL=float(post@c/c.sum()),delta_pre=float((pre-par)@c/c.sum()),delta_post=float((post-par)@c/c.sum()),reduced_NLL=reduce(post),reduced_delta_pre=reduce(pre-par),reduced_delta_post=reduce(post-par),dropped_sequences=len(drop),dropped_prediction_tokens=int(c[~keep].sum()),exposure=exp,flags=r.get('flags',[])))
            groups.setdefault((item['data'],item['budget'],item['lr'],item['stream_seed']),{})[item['arm']]=r
        except Exception as error:bad.append(dict(run_id=r['run_id'],reason=repr(error),clears_with='Restore valid aligned per-document arrays, audit and exact actual sample traces'))
    pairs=[]
    for key,arms in groups.items():
        data,budget,lr,seed=key;drop=cfg.get('overlap',{}).get(data,{}).get('final_indices',[])
        for name,short,long in [('E','k6','k20'),('E64','k6','k64'),('D64','k20','k64')]:
            if short not in arms or long not in arms:continue
            try:
                full=pair_scores(arms[short],arms[long],repeats=10000 if final else 0)
                reduced=pair_scores(arms[short],arms[long],drop,repeats=10000 if final else 0)
                pairs.append(dict(quantity=name,data=data,budget=budget,lr=lr,seed=seed,full=full,reduced=reduced,difference=reduced['E']-full['E'],audit_available=data in cfg.get('overlap',{})))
            except Exception as error:bad.append(dict(pair=str((key,name)),reason=repr(error)))
    overlap={}
    for data in ('fwe','mix'):
        subset=[p for p in pairs if p['data']==data]
        ok=bool(subset) and data in cfg.get('overlap',{}) and all(abs(p['difference'])<.001 for p in subset)
        overlap[data]=dict(immaterial=ok if subset and data in cfg.get('overlap',{}) else None,primary='full' if ok else 'reduced' if data in cfg.get('overlap',{}) else 'unverified')
    selected=read(root/'lr_selection.json') if (root/'lr_selection.json').exists() else {}
    eta=selected.get('lr');readings={}
    def points(quantity,data='fwe',seed=2030):return sorted([p for p in pairs if p['quantity']==quantity and p['data']==data and p['lr']==eta and p['seed']==seed],key=lambda p:float(p['budget'][:-1]))
    def value(p):return p['reduced' if overlap[p['data']]['primary']=='reduced' else 'full']['E']
    es=points('E');ds=points('D64')
    if es:
        e=value(es[-1]);readings['FE1']=dict(value=e,label='persists' if e>=.008 else 'erased' if abs(e)<.003 else 'reversed' if e<=-.008 else 'weak',budget=es[-1]['budget'],not_converged=changed(value(es[-2]),e) if len(es)>1 else None)
    base={p['budget']:value(p) for p in es}
    if {'0.2B','0.8B'}<=base.keys():
        d=base['0.8B']-base['0.2B'];readings['FE2']=dict(label='flat' if abs(d)<.003 else 'changing',direction='up' if d>0 else 'down' if d<0 else 'equal',difference=d)
    if ds:
        d=value(ds[-1]);readings['FE3']=dict(value=d,label='grows' if d>=.003 else 'saturates' if abs(d)<.003 else 'falls',budget=ds[-1]['budget'],not_converged=changed(value(ds[-2]),d) if len(ds)>1 else None)
    mix={p['budget']:value(p) for p in points('E','mix')};other={p['budget']:value(p) for p in points('E',seed=2031)}
    if '0.2B' in base and '0.2B' in other:readings['data_order_noise']=dict(value=abs(base['0.2B']-other['0.2B']),flag=abs(base['0.2B']-other['0.2B'])>.003)
    if all('0.2B' in x for x in (base,mix,other)):readings['FE4']=mixture_reading(mix['0.2B'],base['0.2B'],other['0.2B'])
    for name,reading in readings.items():
        relevant=[r for r in rows if r['data']=='fwe' or name=='FE4']
        reading['unverified']=bool(bad) or any(r['flags'] or r['exposure']['status']=='unverified' for r in relevant)
    for p in root.glob('**/tests-rank*.json'):tests.extend(dict(c,path=str(p)) for c in read(p)['checks'])
    for p in [root/'preparation_checks.json',root/'coordinator_checks.json']:
        if p.exists():tests.extend(read(p)['checks'])
    missing=[e['name'] for e in events if e['returncode'] or (e['module'].endswith('.heal') and not (root/'heals'/e['name']/'result.json').exists())]
    if final:
        for rel in ['reference/dclm/result.json','reference/fwe/result.json',*[f'parents/{a}/result.json' for a in ('k6','k20','k64')]]:
            if not (root/rel).is_file():missing.append(rel)
    decision=dict(final_readout=final,rows=rows,pairs=pairs,readings=readings,overlap=overlap,F0=f0_readout(cfg,root),missing=missing,invalid=bad,flags=[t for t in tests if not t['passed']],input_flags=cfg.get('input_flags',[]),lr_selection=selected,events=events,coverage_complete=final and not missing and not bad,interval_scope='Evaluation-set noise only; no training-seed uncertainty',matched_lr=cfg['peak_lr']/4)
    write(root/'decision_family.json',decision);write(root/'tests.json',dict(checks=tests));write(root/'overlap_audit/readout.json',dict(runs=[dict(run_id=r['run_id'],exposure=r['exposure']) for r in rows],pairs=pairs,overlap=overlap))
    lines=['# A2 Family v1.3','',f'Spec SHA256: {SPEC_SHA}','',f'Final readout: {final}; coverage: {decision["coverage_complete"]}.','']
    for name,value in readings.items(): lines.append(f'- {name}: {value}')
    lines += ['', '| Run | Parent NLL | NLL | Delta pre | Delta post | Exposure | Flags |','|---|---:|---:|---:|---:|---|---:|']
    lines += [f'| {r["run_id"]} | {r["parent"]:.9f} | {r["NLL"]:.9f} | {r["delta_pre"]:.9f} | {r["delta_post"]:.9f} | {r["exposure"]["status"]} | {len(r["flags"])} |' for r in rows]
    lines += ['', '| Contrast / data / budget / LR / order | Full | Reduced | Reduced minus full | Primary | 95% CI | E_pre | G | E/G | E/G 95% CI | Dropped docs / targets |', '|---|---:|---:|---:|---|---|---:|---:|---:|---|---|']
    for p in pairs:
        basis=overlap[p['data']]['primary'];v=p['reduced' if basis=='reduced' else 'full'];ci=v.get('bootstrap',{})
        label=f'{p["quantity"]} / {p["data"]} / {p["budget"]} / {p["lr"]:.8g} / {p["seed"]}'
        lines.append(f'| {label} | {p["full"]["E"]:.9f} | {p["reduced"]["E"]:.9f} | {p["difference"]:+.9f} | {basis} | {ci.get("E_ci95")} | {v["E_pre"]:.9f} | {v["G"]:.9f} | {v["E_over_G"]} | {ci.get("E_over_G_ci95")} | {p["reduced"]["dropped_sequences"]} / {p["reduced"]["dropped_prediction_tokens"]} |')
    lines += ['', f'Matched LR (peak/4): {decision["matched_lr"]}. OpenSci stable E/G: DCLM 0.51, FWE 0.81; annealed: 0.41, 0.50.', '']
    for dataset,ref in decision['F0'].items():
        lines.append(f'F0 {dataset}: { {k:v for k,v in ref.items() if k!="students"} }')
        lines += ['', '| Student | Heal budget | NLL | Student minus 1.3B | Total training tokens |','|---|---|---:|---:|---:|']
        for r in ref.get('students',[]):lines.append(f'| {r["arm"]} | {r["budget"]} | {r["nll"]:.9f} | {r["student_minus_1_3B"]:+.9f} | {r["total_tokens"]:.0f} |')
        lines.append('')
    extensions=read(root/'extensions.json') if (root/'extensions.json').exists() else None
    lines += [f'Conditional extensions and trigger evidence: {extensions}', '', 'Paired document bootstrap: 10,000 resamples, seed 0, token-weighted NLL. Evaluation noise only.', 'Mixture sampling/source qualifications apply to mixture rows. Shared stable-parent caveat: healing also anneals the student.', f'Input qualifications: {cfg.get("input_flags",[])}', '',f'Missing: {missing}; invalid: {bad}.','Remaining: verify backup/native handoff and resume Depth.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return decision
