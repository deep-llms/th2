"""Descriptive F9 readout; all intervals jointly resample matched task items."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import numpy as np
from eval.a2_family_f9 import read,write,items,joint_draws,interval

def summary(p,d):
    return interval(p,d)

def linear(names,draws,weights):
    w=np.array([weights.get(n,0.) for n in names])
    return {t:(p@w,d@w) for t,(p,d) in draws.items()}

def summarize(values):
    return {t:summary(p,d) for t,(p,d) in values.items()}

def ratio(num,den):
    out={}
    for t,(p,d) in num.items():
        q,e=den[t];ci=interval(q,e,scale=1)['ci95']
        out[t]=(interval(p/q,d/e,scale=1) if (ci[0]>0 or ci[1]<0) and np.all(e!=0) else None)
    return out

def analyze(entries,replays):
    byid={e['model_id']:e for e in entries};records={k:items(v) for k,v in replays.items()}
    models={k:{t:float(np.mean(v['values'])) for t,v in r.items()} for k,r in records.items()}
    for row in models.values():
        row['MC8']=sum(v for t,v in row.items() if t!='lambada_openai')/8
        row['MC7']=sum(row[t] for t in records[next(iter(records))] if t not in ('lambada_openai','boolq'))/7
    for model,row in models.items():
        row['LAMBADA_perplexity']=replays[model]['results']['lambada_openai'].get('perplexity,none')
    grouped=defaultdict(lambda:defaultdict(list))
    keys=('family','dataset','cut','variant','budget','lr','seed')
    for e in entries:
        if e['model_id'] in records and e['role']=='student':
            key=tuple(e.get(k) for k in keys)
            grouped[key][e['arm']].append(e)
    lead=[];ainit=[];unavailable=[];pair_inputs={}
    for key,arms in grouped.items():
        conditions=dict(zip(keys,key))
        pairs=[(short,long,a,b) for short,long in [('P300','P1T'),('S72','S188'),('k6','k20'),('k6','k64'),('k20','k64')]
               for a,b in product(arms.get(short,[]),arms.get(long,[]))]
        for short,long,a,b in pairs:
            ids=[a.get('parent_id'),b.get('parent_id'),a['model_id'],b['model_id']]
            if any(i not in records for i in ids):
                unavailable.append(dict(condition=conditions,pair=[short,long],reason='missing parent scores'));continue
            names,draws=joint_draws({i:records[i] for i in ids})
            sp,lp,ss,ls=ids
            gap=linear(names,draws,{lp:1,sp:-1});studentgap=linear(names,draws,{ls:1,ss:-1})
            ev=linear(names,draws,{lp:1,ls:-1,sp:-1,ss:1})
            sn,ln=byid[sp].get('final_nll'),byid[lp].get('final_nll')
            an,bn=a.get('final_nll'),b.get('final_nll')
            g=sn-ln if sn is not None and ln is not None else None
            E=(bn-ln)-(an-sn) if all(x is not None for x in (sn,ln,an,bn)) else None
            er=summarize(ev)
            lead.append(dict(**conditions,pair=[short,long],models=ids,E_nats=E,G_nats=g,
                NLL_kept_share=(g-E)/g if g and E is not None else None,E_down=er,
                accuracy_kept_share=ratio(studentgap,gap),parent_gap=summarize(gap),
                reading={t:'detected' if r['ci95'][0]>0 or r['ci95'][1]<0 else 'not detected' for t,r in er.items()},
                scope='per dataset',flags=[*a.get('flags',[]),*b.get('flags',[])]))
            pair_inputs[(key,short,long,a['model_id'],b['model_id'])]=ids
        if 'P300' in arms and 'X72' in arms:
            for a,b in product(arms['P300'],arms['X72']):
                ids=[a['model_id'],b['model_id']]
                names,draws=joint_draws({i:records[i] for i in ids})
                ainit.append(dict(**conditions,models=ids,A_init_down=summarize(linear(names,draws,{ids[0]:1,ids[1]:-1}))))
    pooled=[]
    candidates=[]
    for (key,short,long,_,_),ids in pair_inputs.items():
        if key[1]!='dclm':continue
        other=tuple('fwe' if i==1 else v for i,v in enumerate(key))
        for (otherkey,s,l,_,_),partner in pair_inputs.items():
            if (otherkey,s,l)==(other,short,long):candidates.append((key,short,long,ids,partner))
    for key,short,long,ids,partner in candidates:
        allids=ids+partner;names,draws=joint_draws({i:records[i] for i in allids})
        weights={}
        for quartet in (ids,partner):
            for i,w in zip(quartet,[-.5,.5,.5,-.5]):weights[i]=weights.get(i,0)+w
        ev=summarize(linear(names,draws,weights))
        pg={};sg={}
        for sp,lp,ss,ls in (ids,partner):
            pg[lp]=.5;pg[sp]=-.5;sg[ls]=.5;sg[ss]=-.5
        constituents=[r for r in lead if r['models'] in (ids,partner)]
        E=sum(r['E_nats'] for r in constituents)/2 if all(r['E_nats'] is not None for r in constituents) else None
        G=sum(r['G_nats'] for r in constituents)/2 if all(r['G_nats'] is not None for r in constituents) else None
        pooled.append(dict(**dict(zip(keys,tuple('pooled' if i==1 else v for i,v in enumerate(key)))),
            models=allids,datasets=['dclm','fwe'],pair=[short,long],E_down=ev,E_nats=E,G_nats=G,
            NLL_kept_share=(G-E)/G if G and E is not None else None,
            parent_gap=summarize(linear(names,draws,pg)),accuracy_kept_share=ratio(linear(names,draws,sg),linear(names,draws,pg)),
            reading={t:'detected' if r['ci95'][0]>0 or r['ci95'][1]<0 else 'not detected' for t,r in ev.items()},
            bootstrap='same joint item indices for both datasets'))
    # Parent anneal gains, and ratios with the correctly joint denominator.
    gains={};parentsets={}
    for ds in ('dclm','fwe','nem'):
        arms={e['arm']:e['model_id'] for e in entries if e['role']=='parent' and e['family']=='opensci'
              and e['dataset']==ds and e['model_id'] in records}
        if not all(a in arms for a in ('S72','S188','P300','P1T')):continue
        parentsets[ds]=arms;names,draws=joint_draws({i:records[i] for i in arms.values()})
        sw={arms['P300']:1,arms['S72']:-1};lw={arms['P1T']:1,arms['S188']:-1}
        dw={**lw,**{i:-w for i,w in sw.items()}}
        nll=[byid[arms[a]].get('final_nll') for a in ('S72','P300','S188','P1T')]
        gn=(nll[2]-nll[3])-(nll[0]-nll[1]) if all(v is not None for v in nll) else None
        gains[ds]=dict(g_short=summarize(linear(names,draws,sw)),g_long=summarize(linear(names,draws,lw)),
                       delta_g=summarize(linear(names,draws,dw)),delta_g_NLL=gn)
        for row in lead:
            if row['dataset']!=ds or row['pair']!=['P300','P1T']:continue
            ids=row['models'];allids=set(ids)|set(arms.values())
            names,draws=joint_draws({i:records[i] for i in sorted(allids)})
            ev=linear(names,draws,dict(zip(ids,[-1,1,1,-1])))
            row['E_down_over_delta_g']=ratio(ev,linear(names,draws,dw))
            row['E_over_delta_g_NLL']=row['E_nats']/gn if gn and row['E_nats'] is not None else None
    if all(ds in parentsets for ds in ('dclm','fwe')):
        allids=[i for ds in ('dclm','fwe') for i in parentsets[ds].values()]
        names,draws=joint_draws({i:records[i] for i in allids});sw={};lw={}
        for ds in ('dclm','fwe'):
            a=parentsets[ds];sw.update({a['P300']:.5,a['S72']:-.5});lw.update({a['P1T']:.5,a['S188']:-.5})
        dw={**lw,**{i:-w for i,w in sw.items()}}
        gains['pooled']=dict(g_short=summarize(linear(names,draws,sw)),g_long=summarize(linear(names,draws,lw)),
            delta_g=summarize(linear(names,draws,dw)),delta_g_NLL=(gains['dclm']['delta_g_NLL']+gains['fwe']['delta_g_NLL'])/2
            if all(gains[ds]['delta_g_NLL'] is not None for ds in ('dclm','fwe')) else None)
        for row in pooled:
            if row['pair']!=['P300','P1T']:continue
            modelids=set(row['models'])|set(allids)
            names,draws=joint_draws({i:records[i] for i in sorted(modelids)})
            ew={}
            for i,w in zip(row['models'],[-.5,.5,.5,-.5]*2):ew[i]=ew.get(i,0)+w
            row['E_down_over_delta_g']=ratio(linear(names,draws,ew),linear(names,draws,dw))
            gn=gains['pooled']['delta_g_NLL']
            row['E_over_delta_g_NLL']=row['E_nats']/gn if gn and row['E_nats'] is not None else None
    reference=[]
    for e in entries:
        if e['role']!='student' or e['model_id'] not in records:continue
        refs=[r for r in entries if r['role']=='reference' and r['dataset']==e['dataset'] and r['model_id'] in records]
        for ref in refs:
            ids=[e['model_id'],ref['model_id']];names,draws=joint_draws({i:records[i] for i in ids})
            reference.append(dict(student=ids[0],reference=ids[1],difference=summarize(linear(names,draws,{ids[0]:1,ids[1]:-1}))))
    drops=[]
    for e in entries:
        if e['role']!='student' or e['model_id'] not in records or e.get('parent_id') not in records:continue
        parent,student=e['parent_id'],e['model_id']
        drops.append(dict(student=student,parent=parent,
            drop_pp={t:100*(models[parent][t]-models[student][t]) for t in models[parent] if t!='LAMBADA_perplexity'}))
    trends=[];series=defaultdict(list)
    for row in lead:
        # Admission reruns do not replace the registered original endpoint in
        # a budget series. Both remain individually visible in the lead table.
        if any(byid[i]['part'] in ('FT9','YT1') for i in row['models'][2:] if 'part' in byid[i]):continue
        key=tuple(row[k] for k in ('family','dataset','cut','variant','lr','seed'))+tuple(row['pair'])
        series[key].append(row)
    for key,rows in series.items():
        rows.sort(key=lambda r:r['budget'])
        if len(rows)<3:continue
        if len({r['budget'] for r in rows})!=len(rows):
            unavailable.append(dict(series=list(key),reason='Multiple reruns at one budget; no arbitrary lineage selected for trend'))
            continue
        ids=sorted({i for r in rows for i in r['models']});names,draws=joint_draws({i:records[i] for i in ids})
        x=np.log([r['budget'] for r in rows]);w=(x-x.mean())/np.sum((x-x.mean())**2)
        weights={}
        for r,c in zip(rows,w):
            for i,a in zip(r['models'],[-1,1,1,-1]):weights[i]=weights.get(i,0)+c*a
        trends.append(dict(series=list(key),budgets=[r['budget'] for r in rows],slope_pp_per_log_token_budget=summarize(linear(names,draws,weights))))
    # Descriptive through-origin calibration: drop(P300) = coefficient * delta_NLL.
    # Explicitly record estimator because the spec does not choose an intercept.
    consistency=[]
    for key,rows in series.items():
        if list(key[-2:])!=['P300','P1T']:continue
        usable=[r for r in rows if r['E_nats'] is not None and all(byid[i].get('final_nll') is not None for i in r['models'])]
        if len(usable)<2:continue
        x=np.array([byid[r['models'][2]]['final_nll']-byid[r['models'][0]]['final_nll'] for r in usable])
        if np.sum(x*x)==0:continue
        coefficients={t:float(x@np.array([models[r['models'][0]][t]-models[r['models'][2]][t] for r in usable])/(x@x))
                      for t in models[usable[0]['models'][0]] if t!='LAMBADA_perplexity'}
        consistency.append(dict(series=list(key),estimator='least-squares through origin across available budgets',
            drop_per_nat=coefficients,rows=[dict(budget=r['budget'],predicted_E_down_pp={t:100*c*r['E_nats'] for t,c in coefficients.items()},measured=r['E_down']) for r in usable]))
    return dict(models=models,lead=lead,pooled=pooled,anneal_gains=gains,A_init_down=ainit,drops=drops,
                reference=reference,budget_trends=trends,NLL_consistency=consistency,unavailable=unavailable,
                units='accuracy contrasts percentage points; NLL nats; kept shares fractions',resamples=10000,seed=0,
                trend_policy='Original registered endpoints; FT9/YT1 admission reruns shown separately, excluded from budget regression')

def render(result):
    lines=['## Family v1.9 — F9 downstream','',
           'Descriptive only; original F0–F8 and Depth readings unchanged. Intervals cover evaluation items.',
           '', '| Dataset | Pair | Cut | Variant | Budget | LR | Seed | E (nats) | E_down MC8 (pp, CI95) | E_down LAMBADA (pp, CI95) | NLL kept | Accuracy kept MC8 | Accuracy kept LAMBADA | Students |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    def val(v):return 'N/A' if v is None else (f"{v['estimate']:.6f} [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]" if isinstance(v,dict) else str(v))
    for r in result['pooled']+result['lead']:
        lines.append('| '+' | '.join(str(v) for v in [r['dataset'],'/'.join(r['pair']),r['cut'],r['variant'],r['budget'],r['lr'],r['seed'],val(r['E_nats']),val(r['E_down']['MC8']),val(r['E_down']['lambada_openai']),val(r['NLL_kept_share']),val(r['accuracy_kept_share']['MC8']),val(r['accuracy_kept_share']['lambada_openai']),', '.join(r['models'][2:4])])+' |')
    lines+=['','### Anneal gains','',
        '| Dataset | Metric | g_short (pp, CI95) | g_long (pp, CI95) | Delta g (pp, CI95) | Delta g NLL |',
        '|---|---|---|---|---|---|']
    for ds,r in result['anneal_gains'].items():
        for t in ('MC8','MC7','lambada_openai'):
            lines.append('| '+' | '.join([ds,t,val(r['g_short'][t]),val(r['g_long'][t]),val(r['delta_g'][t]),val(r['delta_g_NLL'])])+' |')
    lines+=['','### X72 contrast','', '| Dataset | Cut | Budget | MC8 (pp, CI95) | LAMBADA (pp, CI95) |', '|---|---|---|---|---|']
    for r in result['A_init_down']:
        lines.append('| '+' | '.join(map(str,[r['dataset'],r['cut'],r['budget'],val(r['A_init_down']['MC8']),val(r['A_init_down']['lambada_openai'])]))+' |')
    lines+=['','### Per-task length contrasts','', '| Dataset | Pair | Cut | Budget | LR | Seed | Task | E_down (pp, CI95) |', '|---|---|---|---|---|---|---|---|']
    for r in result['pooled']+result['lead']:
        for t,v in r['E_down'].items():
            lines.append('| '+' | '.join(map(str,[r['dataset'],'/'.join(r['pair']),r['cut'],r['budget'],r['lr'],r['seed'],t,val(v)]))+' |')
    lines+=['','### Model accuracies and LAMBADA perplexity','']
    if result['models']:
        tasks=list(next(iter(result['models'].values())))
        lines+=['| Model | '+' | '.join(tasks)+' |','|---|'+'---|'*len(tasks)]
        for model,row in result['models'].items():
            lines.append('| '+model+' | '+' | '.join(val(row[t]) for t in tasks)+' |')
    lines+=['','### Student minus 1.3B reference','', '| Student | Reference | MC8 (pp, CI95) | LAMBADA (pp, CI95) |', '|---|---|---|---|']
    for r in result['reference']:
        lines.append('| '+' | '.join([r['student'],r['reference'],val(r['difference']['MC8']),val(r['difference']['lambada_openai'])])+' |')
    lines+=['','### Descriptive readings','', '| Dataset | Pair | Budget | Cut | LR | Seed | MC8 | LAMBADA |', '|---|---|---|---|---|---|---|---|']
    for r in result['pooled']+[r for r in result['lead'] if r['dataset']=='dclm' and r['budget']==12.8]:
        lines.append('| '+' | '.join(map(str,[r['dataset']+(' only' if r['budget']==12.8 else ''),'/'.join(r['pair']),r['budget'],r['cut'],r['lr'],r['seed'],r['reading']['MC8'],r['reading']['lambada_openai']]))+' |')
    lines+=['','### Budget trends','',result['trend_policy'],'', '| Series | Budgets | MC8 slope (pp/log budget, CI95) | LAMBADA slope |', '|---|---|---|---|']
    for r in result['budget_trends']:
        lines.append('| '+' | '.join(map(str,[r['series'],r['budgets'],val(r['slope_pp_per_log_token_budget']['MC8']),val(r['slope_pp_per_log_token_budget']['lambada_openai'])]))+' |')
    lines+=['','### NLL prediction against measured accuracy penalty','', '| Series | Budget | Metric | Predicted E_down (pp) | Measured E_down (pp, CI95) |', '|---|---|---|---|---|']
    for series in result['NLL_consistency']:
        for r in series['rows']:
            for t in ('MC8','lambada_openai'):
                lines.append('| '+' | '.join(map(str,[series['series'],r['budget'],t,r['predicted_E_down_pp'][t],val(r['measured'][t])]))+' |')
    lines+=['','### Penalty relative to anneal gain','', '| Dataset | Pair | Cut | Budget | E / Delta g NLL | E_down / Delta g MC8 | E_down / Delta g LAMBADA |', '|---|---|---|---|---|---|---|']
    for r in result['pooled']+result['lead']:
        if 'E_down_over_delta_g' in r:
            lines.append('| '+' | '.join(map(str,[r['dataset'],'/'.join(r['pair']),r['cut'],r['budget'],val(r['E_over_delta_g_NLL']),val(r['E_down_over_delta_g']['MC8']),val(r['E_down_over_delta_g']['lambada_openai'])]))+' |')
    lines+=['','### Parent minus student accuracy','', '| Student | Parent | MC8 drop (pp) | MC7 drop (pp) | LAMBADA drop (pp) |', '|---|---|---|---|---|']
    for r in result['drops']:
        lines.append('| '+' | '.join(map(str,[r['student'],r['parent'],r['drop_pp']['MC8'],r['drop_pp']['MC7'],r['drop_pp']['lambada_openai']]))+' |')
    lines+=['','Full joint-bootstrap MC7, budget slopes, NLL calibration, gain ratios and flags: contrasts.json.',
        'A missing kept share means the parent-gap CI includes zero, or a resampled denominator is zero.', '']
    inventory=result.get('inventory',{})
    lines+=['### Coverage and provenance','',
        'Available models and exact weight/source hashes: preparation/FT12_inventory.json. All missing or failed models below remain unscored; no retraining.', '']
    for category in ('missing','hash_failures','unclassified'):
        for row in inventory.get(category,[]):lines.append('- '+category+': '+str(row))
    lines+=['','Earlier downstream results for missing weights (separate origin, excluded from fresh F9 contrasts):','']
    for row in result.get('earlier_downstream',[]):lines.append('- '+str(row))
    lines+=['','Flags retained (including numerical batch differences):','']
    for row in result.get('flags',[]):lines.append('- '+str(row))
    return '\n'.join(lines)
