"""F10 item-paired contrasts and additive readout, computed only on CPU."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import csv
import numpy as np
from eval.a2_family_f10 import write,SPEC_SHA
from eval.a2_family_f9 import read
from eval.a2_request_bridge import digest
from eval.downstream_suite import COUNTS
from scripts.a2_family_f9_v110_run import locked_report


def interval(point,draws):
    return dict(estimate=float(point),ci95=np.quantile(draws,[.025,.975]).tolist())


def ratio(numerator,denominator):
    p,d=numerator;q,e=denominator;lo,hi=np.quantile(e,[.025,.975])
    return interval(p/q,d/e) if q and (lo>0 or hi<0) and np.all(e!=0) else None


def sample(records):
    """Reuse F9's exact sorted-task multinomial stream for the original nine.

    New tasks use a separate seed-0 stream, keeping the F9 draws unchanged.
    Each task uses identical item indices across all available models.
    """
    result={}
    tasks={t for r in records.values() for t in r}
    for block in ([t for t in sorted(COUNTS) if t in tasks],sorted(tasks-set(COUNTS))):
        rng=np.random.default_rng(0)
        for task in block:
            names=[m for m,r in records.items() if task in r]
            rows=[records[m][task] for m in names]
            if len({r['identity'] for r in rows})!=1:raise ValueError('Unpaired task identities')
            matrix=np.asarray([r['values'] for r in rows],dtype=float).T;n=len(matrix)
            if not n or not np.isfinite(matrix).all():raise ValueError('Invalid item values')
            boots=[]
            for start in range(0,10000,64):
                weights=rng.multinomial(n,np.full(n,1/n),size=min(64,10000-start))
                boots.append(weights@matrix/n)
            result[task]=(names,matrix.mean(0),np.concatenate(boots))
    for label,ts in [('MC8',[t for t in COUNTS if t!='lambada_openai']),('MC7',[t for t in COUNTS if t not in ('lambada_openai','boolq')])]:
        if all(t in result for t in ts):
            names=[m for m in records if all(m in result[t][0] for t in ts)]
            if names:
                p=sum(result[t][1][[result[t][0].index(m) for m in names]] for t in ts)/len(ts)
                d=sum(result[t][2][:,[result[t][0].index(m) for m in names]] for t in ts)/len(ts)
                result[label]=(names,p,d)
    return result


def linear(sampled,weights,scale=1.):
    out={}
    for task,(names,p,d) in sampled.items():
        if not set(weights)<=set(names):continue
        w=np.asarray([weights.get(n,0.) for n in names])*scale
        out[task]=(float(p@w),d@w)
    return out


def pairs(entries):
    parents={e['model_id']:e for e in entries if e['role']=='parent'}
    groups=defaultdict(dict)
    for e in entries:
        if e['role'] not in ('student','oneshot') or e['arm']=='X72':continue
        key=(e['family'],e['dataset'],e['budget'])
        if e['arm'] in groups[key]:raise ValueError('Ambiguous endpoint lineage')
        groups[key][e['arm']]=e
    rows=[]
    for (family,dataset,budget),arms in groups.items():
        for short,long in [('P300','P1T'),('S72','S188'),('k6','k20'),('k6','k64'),('k20','k64')]:
            if short not in arms or long not in arms:continue
            a,b=arms[short],arms[long];ids=[a['parent_id'],b['parent_id'],a['model_id'],b['model_id']]
            if not all(m in parents for m in ids[:2]):continue
            pa,pb=parents[ids[0]],parents[ids[1]]
            nll=[e.get('final_nll') for e in [pa,pb,a,b]]
            g=nll[0]-nll[1] if all(x is not None for x in nll) else None
            E=nll[3]-nll[1]-nll[2]+nll[0] if g is not None else None
            rows.append(dict(family=family,dataset=dataset,budget_B=budget,pair=[short,long],models=ids,
                E_NLL=E,NLL_parent_gap=g,NLL_kept_share=(g-E)/g if g else None))
    pooled=[]
    for row in rows:
        if row['dataset']!='dclm':continue
        match=[r for r in rows if r['dataset']=='fwe' and all(r[k]==row[k] for k in ('family','pair','budget_B'))]
        if len(match)!=1:continue
        other=match[0];g=(row['NLL_parent_gap']+other['NLL_parent_gap'])/2
        E=(row['E_NLL']+other['E_NLL'])/2
        pooled.append(dict(row,dataset='pooled',models=row['models']+other['models'],
                           E_NLL=E,NLL_parent_gap=g,NLL_kept_share=(g-E)/g if g else None))
    return rows+pooled


def contrast_readout(entries,records,scale=1.):
    sampled=sample(records);rows=pairs(entries);internal={}
    for index,row in enumerate(rows):
        ids=row['models'];factor=4/len(ids);weights={m:w*factor for m,w in zip(ids,[-1,1,1,-1]*(len(ids)//4))}
        E=linear(sampled,weights,scale);pg={};sg={}
        for start in range(0,len(ids),4):
            sp,lp,ss,ls=ids[start:start+4];pg.update({sp:-factor,lp:factor});sg.update({ss:-factor,ls:factor})
        parentgap=linear(sampled,pg,scale);studentgap=linear(sampled,sg,scale)
        row.update(E={t:interval(*v) for t,v in E.items()},
                   parent_gap={t:interval(*v) for t,v in parentgap.items()},
                   kept_share={t:ratio(studentgap[t],parentgap[t]) for t in E},
                   reading={t:'detected' if min(interval(*v)['ci95'])>0 or max(interval(*v)['ci95'])<0 else 'not detected' for t,v in E.items()})
        internal[index]=E
    series=defaultdict(list)
    for i,row in enumerate(rows):
        if row['budget_B']>0:series[(row['family'],row['dataset'],tuple(row['pair']))].append(i)
    trends=[]
    for key,indices in series.items():
        if len(indices)<3:continue
        indices.sort(key=lambda i:rows[i]['budget_B']);budgets=[rows[i]['budget_B'] for i in indices]
        x=np.log(budgets);w=(x-x.mean())/sum((x-x.mean())**2)
        tasks=set.intersection(*(set(internal[i]) for i in indices))
        trends.append(dict(family=key[0],dataset=key[1],pair=key[2],budgets_B=budgets,
             slope_per_log_budget={t:interval(sum(c*internal[i][t][0] for c,i in zip(w,indices)),sum(c*internal[i][t][1] for c,i in zip(w,indices))) for t in tasks}))
    gains={}
    for dataset in ('dclm','fwe','pooled'):
        ds=['dclm','fwe'] if dataset=='pooled' else [dataset];sw={};lw={};valid=True
        for d in ds:
            arm={e['arm']:e['model_id'] for e in entries if e['role']=='parent' and e['family']=='opensci' and e['dataset']==d}
            if not {'P300','P1T','S72','S188'}<=set(arm):valid=False;break
            f=1/len(ds);sw.update({arm['P300']:f,arm['S72']:-f});lw.update({arm['P1T']:f,arm['S188']:-f})
        if valid:
            dw={**lw,**{k:-v for k,v in sw.items()}}
            gains[dataset]={name:{t:interval(*v) for t,v in linear(sampled,w,scale).items()} for name,w in [('g_short',sw),('g_long',lw),('delta_g',dw)]}
    drops=[]
    for e in entries:
        parent=e.get('parent_id');model=e['model_id']
        if parent in records and model in records:
            drops.append(dict(model=model,parent=parent,drop={t:interval(*v) for t,v in linear(sampled,{parent:1,model:-1},scale).items()}))
    return dict(contrasts=rows,trends=trends,anneal_gains=gains,drops=drops,
                models={m:{t:float(np.mean(r['values'])) for t,r in tasks.items()} for m,tasks in records.items()}),sampled


def confidence_items(exported,replay):
    """Character-normalized gold score from the exact F9 likelihood requests."""
    result={}
    for task,rows in exported['tasks'].items():
        refs={r['doc_id']:r for r in replay['samples'][task]};values=[]
        for row in rows:
            if task=='lambada_openai':
                target=refs[row['doc_id']]['target'];length=len(target.lstrip())
                if not length:raise ValueError('Empty LAMBADA answer')
                values.append(-row['target_NLL']/length)
            else:
                g=row['gold_choice']
                length=len(refs[row['doc_id']]['arguments'][g][1].lstrip()) if task=='winogrande' else row['choice_character_lengths'][g]
                if length<=0:raise ValueError('Empty gold continuation')
                values.append(row['raw_loglikelihoods'][g]/length)
        result[task]=dict(identity=exported['continuous_records'][task]['identity'],values=values,
                          correct=[r['correctness'] for r in rows])
    return result


def cohort_bootstrap(values, correct, rng, resamples=10000):
    """Fix each four-model cohort; jointly resample their eligible union.

    For a pooled row, a document can belong to one or both fixed corpora
    cohorts. Shared union counts preserve covariance on overlapping items;
    each corpus mean is normalized by its own in-cohort bootstrap count.
    No requirement that all eight models are correct is introduced.
    """
    values=np.asarray(values,float);correct=np.asarray(correct)
    if values.shape!=correct.shape or values.shape[1] not in (4,8):raise ValueError('Invalid cohort matrix')
    masks=np.stack([np.all(correct[:,i:i+4]==1,axis=1) for i in range(0,values.shape[1],4)],axis=1)
    counts=masks.sum(0)
    if np.any(counts==0):return None,None,counts.tolist()
    union=masks.any(1);mask=masks[union].astype(float)
    differences=np.stack([values[union,i:i+4]@np.array([-1,1,1,-1]) for i in range(0,values.shape[1],4)],axis=1)*mask
    point=float(np.mean(differences.sum(0)/counts));n=int(union.sum());boots=[];total=0
    while total<resamples:
        w=rng.multinomial(n,np.full(n,1/n),size=min(64,resamples-total));den=w@mask
        ok=(den>0).all(1)
        b=np.mean((w@differences)[ok]/den[ok],axis=1)
        boots.append(b);total+=len(b)
    return point,np.concatenate(boots),counts.tolist()


def restricted_contrasts(entries,records):
    rows=[]
    for row in pairs(entries):
        ids=row['models'];tasks=set.intersection(*(set(records.get(m,{})) for m in ids))
        summaries={};draw_by_task={};counts={};rng=np.random.default_rng(0)
        for task in sorted(tasks):
            if any(records[m][task].get('correct') is None for m in ids):continue
            values=[records[m][task] for m in ids]
            if len({r['identity'] for r in values})!=1:raise ValueError('Unpaired restricted cohort')
            point,draws,ns=cohort_bootstrap(np.asarray([r['values'] for r in values]).T,
                                         np.asarray([r['correct'] for r in values]).T,rng)
            counts[task]=dict(zip(['dclm','fwe'] if len(ids)==8 else [row['dataset']],ns))
            summaries[task]=interval(point,draws) if point is not None else None
            if point is not None:draw_by_task[task]=(point,draws)
        for label,ts in [('MC8',[t for t in COUNTS if t!='lambada_openai']),('MC7',[t for t in COUNTS if t not in ('lambada_openai','boolq')])]:
            if all(t in draw_by_task for t in ts):
                summaries[label]=interval(sum(draw_by_task[t][0] for t in ts)/len(ts),sum(draw_by_task[t][1] for t in ts)/len(ts))
        rows.append(dict(row,E_conf=summaries,counts=counts,cohort_rule='all four correct per corpus; pooled equal-corpus mean',
            reading={t:'detected' if r and (r['ci95'][0]>0 or r['ci95'][1]<0) else 'not detected' if r else 'empty cohort' for t,r in summaries.items()}))
    return rows


def replay_new(prepared,scores,values,generation=None):
    """Independent reconstruction from saved raw LL and decoded strings."""
    from eval.a2_family_f10_tasks import answer_metrics
    if scores['prepared_sha256']!=digest(prepared) or scores['request_sha256']!=prepared['request_sha256']:
        raise ValueError('New raw request identity mismatch')
    for task,rows in prepared['tasks'].items():
        actual=values['tasks'][task]
        if len(rows)!=len(actual):raise ValueError('Missing likelihood items')
        for row,saved in zip(rows,actual,strict=True):
            ll=np.array([scores['scores'][i][0] for i in row['requests']]);chars=np.array(row['character_lengths']);gold=row['gold']
            if not np.isfinite(ll).all() or saved['doc_id']!=row['doc_id']:raise ValueError('Invalid raw LL or item order')
            if saved['raw_loglikelihoods']!=ll.tolist() or saved['character_lengths']!=chars.tolist():raise ValueError('Raw LL export mismatch')
            if abs(saved['conf']-ll[gold]/chars[gold])>1e-12:raise ValueError('Confidence replay mismatch')
            if task=='mmlu':
                norm=ll/chars;prob=np.exp(norm-norm.max());prob/=prob.sum()
                if saved['correctness']!=int(norm.argmax()==gold) or saved['acc']!=int(ll.argmax()==gold) or abs(saved['p_correct']-prob[gold])>1e-12:
                    raise ValueError('MMLU replay mismatch')
    if generation is not None:
        rows=prepared['tasks']['triviaqa'];actual=generation['items']
        if len(rows)!=len(actual) or generation['task_identity']!=prepared['task_identity']['triviaqa']:raise ValueError('Generation coverage mismatch')
        for row,saved in zip(rows,actual,strict=True):
            m=answer_metrics(saved['text'],row['aliases'])
            if row['doc_id']!=saved['doc_id'] or any(abs(saved[k]-v)>1e-12 for k,v in m.items()):raise ValueError('Generation replay mismatch')
        for key,column in [('exact_match','exact_match'),('token_f1','token_f1')]:
            if abs(generation[key]-np.mean([r[column] for r in actual]))>1e-12:raise ValueError('Generation aggregate mismatch')
    return True


def report(cfg,root):
    old=Path(cfg['f9_data_root']);original=read(old/'contrasts.json')['model_metadata']
    from efficient_llm.diagnostic_data import sha256
    for name,item in read(old/'backup/verified.json')['files'].items():
        if name.startswith('per_item/') or name.endswith('replay.json') or name in ('contrasts.json','portable_index.json'):
            if sha256(old/name)!=item['sha256']:raise ValueError('F9 analysis input changed: '+name)
    entries=[e for e in original if e['family']!='opensci13']+cfg['oneshot_entries']
    byid={e['model_id']:e for e in entries};conf={};pc={}
    index=read(old/'portable_index.json')['models']
    for model in byid:
        if model in index:
            exported=read(old/'per_item'/(model+'.json'));replay=read(old/index[model]['replay']['path'])
        elif (root/'nine'/model/'per_item.json').is_file():
            exported=read(root/'nine'/model/'per_item.json');replay=read(root/'nine'/model/'replay.json')
        else:continue
        conf[model]=confidence_items(exported,replay)
        pc[model]={t:dict(identity=exported['continuous_records'][t]['identity'],values=[r['p_correct'] for r in rows])
                   for t,rows in exported['tasks'].items() if t!='lambada_openai'}
    newacc={};newconf={};newmodels={};rawprepared=read(cfg['prepared']['path'])
    for e in sum(cfg['groups'],[]):
        model=e['model_id'];folder=root/'new'/model
        if not (folder/'likelihood_items.json').is_file():continue
        values=read(folder/'likelihood_items.json')
        replay_new(rawprepared,read(folder/'scores.json'),values,read(folder/'generation_items.json') if (folder/'generation_items.json').is_file() else None)
        newconf[model]={};newacc[model]={}
        for task,rows in values['tasks'].items():
            identity=values['task_identity'][task]
            correctness=[r['correctness'] for r in rows] if task=='mmlu' else None
            newconf[model][task]=dict(identity=identity,values=[r['conf'] for r in rows],correct=correctness)
            if task=='mmlu':
                newacc[model][task]=dict(identity=identity,values=correctness)
                subjects=defaultdict(list)
                for r in rows:subjects[r['subject']].append(r['correctness'])
                newmodels[model]=dict(MMLU_acc_norm=float(np.mean(correctness)),MMLU_acc=float(np.mean([r['acc'] for r in rows])),
                                      MMLU_subject_average=float(np.mean([np.mean(v) for v in subjects.values()])))
                pc.setdefault(model,{})[task]=dict(identity=identity,values=[r['p_correct'] for r in rows])
        if (folder/'generation_items.json').is_file():
            gen=read(folder/'generation_items.json');correct=[r['exact_match'] for r in gen['items']]
            if [r['doc_id'] for r in gen['items']]!=[r['doc_id'] for r in rawprepared['tasks']['triviaqa']]:raise ValueError('Generation item order mismatch')
            newconf[model]['triviaqa']['correct']=correct
            newacc[model]['triviaqa']=dict(identity=gen['task_identity'],values=correct)
            newmodels[model].update(TriviaQA_EM=gen['exact_match'],TriviaQA_F1=gen['token_f1'])
        conf.setdefault(model,{}).update(newconf[model])
    newresult,_=contrast_readout(entries,newacc,scale=100.)
    newresult['model_scores']=newmodels
    predictions=[]
    for pair,dataset,budget,direction in [(['P300','P1T'],'pooled',3.2,'positive detected'),(['P300','P1T'],'dclm',12.8,'positive detected'),(['S72','S188'],'pooled',3.2,'not detected')]:
        matches=[r for r in newresult['contrasts'] if r['pair']==pair and r['dataset']==dataset and r['budget_B']==budget]
        v=matches[0]['E'].get('triviaqa') if len(matches)==1 else None
        detected=v is not None and (v['ci95'][0]>0 or v['ci95'][1]<0)
        supported=(v['ci95'][0]>0 if direction=='positive detected' else not detected) if v else None
        predictions.append(dict(pair=pair,dataset=dataset,budget_B=budget,prediction=direction,result=v,supported=supported))
    for r in newresult['trends']:
        if r['pair']==('P300','P1T'):
            v=r['slope_per_log_budget'].get('triviaqa')
            predictions.append(dict(dataset=r['dataset'],pair=r['pair'],prediction='slope CI contains zero',result=v,
                supported=v['ci95'][0]<=0<=v['ci95'][1] if v else None))
    newresult['registered_TriviaQA_predictions']=predictions
    write(root/'F10c.json',newresult)
    confidence,draws=contrast_readout(entries,conf)
    stages=[];joint_correct=[]
    for e in entries:
        if e['role'] not in ('student','oneshot'):continue
        model=e['model_id'];parent=e.get('parent_id');oneshot=str(parent)+'-oneshot'
        if model not in conf or parent not in conf or oneshot not in conf:continue
        for task in sorted(set(conf[model])&set(conf[parent])&set(conf[oneshot])):
            ids=[parent,oneshot,model];vals=[conf[m][task] for m in ids]
            if len({r['identity'] for r in vals})!=1:raise ValueError('Stage items differ')
            means=[float(np.mean(r['values'])) for r in vals]
            pmeans=[float(np.mean(pc[m][task]['values'])) if task in pc.get(m,{}) else None for m in ids]
            stages.append(dict(parent=parent,model=model,oneshot=oneshot,task=task,conf=means,p_correct=pmeans,p_correct_differences=({k:float(v) for k,v in zip(('pruning_loss','recovered','still_missing'),(pmeans[0]-pmeans[1],pmeans[2]-pmeans[1],pmeans[0]-pmeans[2]))} if all(v is not None for v in pmeans) else None),
                pruning_loss=means[0]-means[1],recovered=means[2]-means[1],still_missing=means[0]-means[2]))
            if vals[0].get('correct') is not None and vals[2].get('correct') is not None:
                mask=(np.asarray(vals[0]['correct'])==1)&(np.asarray(vals[2]['correct'])==1)
                joint_correct.append(dict(parent=parent,model=model,task=task,items=int(mask.sum()),
                    parent_conf=float(np.mean(np.asarray(vals[0]['values'])[mask])) if mask.any() else None,
                    student_conf=float(np.mean(np.asarray(vals[2]['values'])[mask])) if mask.any() else None))
    # Equal-task MC means, not an item-count-weighted pool.
    for label,ts in [('MC8',[t for t in COUNTS if t!='lambada_openai']),('MC7',[t for t in COUNTS if t not in ('lambada_openai','boolq')])]:
        for model in {r['model'] for r in stages}:
            group=[r for r in stages if r['model']==model and r['task'] in ts]
            if len(group)!=len(ts):continue
            stages.append(dict(parent=group[0]['parent'],model=model,oneshot=group[0]['oneshot'],task=label,
                conf=np.mean([r['conf'] for r in group],axis=0).tolist(),p_correct=np.mean([r['p_correct'] for r in group],axis=0).tolist(),
                **{k:float(np.mean([r[k] for r in group])) for k in ('pruning_loss','recovered','still_missing')}))
    for r in stages:
        p,o,m=r['parent'],r['oneshot'],r['model'];t=r['task']
        r['intervals']={name:interval(*linear(draws,w)[t]) for name,w in
            [('pruning_loss',{p:1,o:-1}),('recovered',({} if m==o else {m:1,o:-1})),('still_missing',{p:1,m:-1})]}
    for label,ts in [('MC8',[t for t in COUNTS if t!='lambada_openai']),('MC7',[t for t in COUNTS if t not in ('lambada_openai','boolq')])]:
        for m in {r['model'] for r in joint_correct}:
            group=[r for r in joint_correct if r['model']==m and r['task'] in ts]
            if len(group)==len(ts) and all(r['items'] for r in group):
                joint_correct.append(dict(parent=group[0]['parent'],model=m,task=label,items={r['task']:r['items'] for r in group},
                    parent_conf=float(np.mean([r['parent_conf'] for r in group])),student_conf=float(np.mean([r['student_conf'] for r in group]))))
    confidence.update(stage_table=stages,parent_student_joint_correct=joint_correct,
                      same_answer_contrasts=restricted_contrasts(entries,conf),
                      conf_definition='gold answer LL / canonical answer character length excluding delimiter; Winogrande scores the canonical gold-conditioned suffix and divides by its characters (leading delimiter excluded); it is a continuation-confidence proxy, not a direct LL of the inserted choice; MC8/MC7 inherit this qualification; LAMBADA excludes leading delimiter whitespace',
                      pooled_correct_cohort_note='Each corpus fixes its all-four-correct cohort. Pooled contrasts average the two corpus means; joint bootstrap resamples their eligible union and normalizes within each fixed cohort, retaining covariance of shared items. No all-eight restriction.')
    write(root/'confidence_items.json',dict(records=conf,p_correct=pc))
    write(root/'F10b.json',confidence)
    tests=read(Path(cfg['f10a_root'])/'tests.json')
    sections=['## Family v1.11 — F10','', 'Spec SHA256: '+SPEC_SHA,'',
              'Evaluation only. Earlier F0–F9 and Depth results and readings remain unchanged. CIs cover evaluation items, not training seeds; joint 10,000 resamples, seed 0.','',
              '### Knowledge tasks: length penalty (accuracy pp) and NLL','',
              '| Dataset | Pair | Budget B | Task | E_down [CI95] | Kept share | NLL E | NLL kept | Reading |',
              '|---|---|---|---|---|---|---|---|---|']
    def fmt(v):
        if v is None:return 'N/A'
        if isinstance(v,dict):return f"{v['estimate']:.6f} [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]"
        return str(v)
    def table(rows,key):
        lines=[]
        for r in rows:
            for task,value in r[key].items():
                lines.append('| '+' | '.join(map(str,[r['dataset'],'/'.join(r['pair']),r['budget_B'],task,fmt(value),
                    fmt(r.get('kept_share',{}).get(task)),fmt(r['E_NLL']),fmt(r['NLL_kept_share']),r.get('reading',{}).get(task,'')]))+' |')
        return lines
    sections+=table(newresult['contrasts'],'E')
    sections+=['','### Parent and model scores','', '| Model | MMLU acc_norm | MMLU acc | Subject average | TriviaQA EM | TriviaQA F1 |','|---|---|---|---|---|---|']
    for m,r in newmodels.items():sections.append('| '+m+' | '+' | '.join(fmt(r.get(k)) for k in ['MMLU_acc_norm','MMLU_acc','MMLU_subject_average','TriviaQA_EM','TriviaQA_F1'])+' |')
    sections+=['','### Confidence length contrasts (nats / character)','', '| Dataset | Pair | Budget B | Task | E_conf [CI95] | Kept share | NLL E | NLL kept | Reading |','|---|---|---|---|---|---|---|---|---|']
    sections+=table(confidence['contrasts'],'E')
    sections+=['','### Same answers, different confidence','',confidence['pooled_correct_cohort_note'],'',
               '| Dataset | Pair | Budget B | Task | Items | E_conf [CI95] | Reading |','|---|---|---|---|---|---|---|']
    for r in confidence['same_answer_contrasts']:
        for t,v in r['E_conf'].items():sections.append('| '+' | '.join(map(str,[r['dataset'],'/'.join(r['pair']),r['budget_B'],t,r['counts'].get(t,'per-task cohorts'),fmt(v),r['reading'].get(t)]))+' |')
    sections+=['','### F10a: tests surviving Holm correction','',
               'Retrospective correction of already reported F9 results; independent families of 36, 27 and 27 tests. Normal-approximation p is corrected by Holm. Bootstrap p uses the centered null distribution with a +1 finite-resample correction.','',
               '| Family | Dataset | Budget B | Task | E_down pp | Adjusted p | Sign |','|---|---|---|---|---|---|---|']
    for family,rs in tests['families'].items():
        for r in rs:
            if r['survives']:sections.append('| '+' | '.join(map(str,[family,r['dataset'],r['budget_B'],r['task'],r['E_down_pp'],r['p_holm'],r['sign']]))+' |')
    sections+=['','### Confidence stage table','', '| Model | Task | Parent conf | One-shot conf | Student conf | Lost | Recovered | Missing |','|---|---|---|---|---|---|---|---|']
    for r in stages:sections.append('| '+' | '.join(map(str,[r['model'],r['task'],*r['conf'],r['pruning_loss'],r['recovered'],r['still_missing']]))+' |')
    sections+=['','### Parent/student means on jointly correct items','',
               '| Model | Task | Items | Parent conf | Student conf |','|---|---|---|---|---|']
    for r in joint_correct:sections.append('| '+' | '.join(map(str,[r['model'],r['task'],r['items'],r['parent_conf'],r['student_conf']]))+' |')
    sections+=['','### Stage probabilities and confidence-difference intervals','',
        '| Model | Task | Parent/one-shot/healed p_correct | Lost CI | Recovered CI | Missing CI |','|---|---|---|---|---|---|']
    for r in stages:sections.append('| '+' | '.join(map(str,[r['model'],r['task'],r['p_correct'],*[fmt(r['intervals'][k]) for k in ('pruning_loss','recovered','still_missing')]]))+' |')
    sections+=['','### Registered TriviaQA predictions','']
    for r in predictions:sections.append('- '+str(r))
    for title,data in [('Knowledge',newresult),('Confidence',confidence)]:
        sections+=['','### '+title+' budget slopes and anneal gains','',
                    '| Dataset | Pair | Task | Slope / log budget CI |','|---|---|---|---|']
        for r in data['trends']:
            for t,v in r['slope_per_log_budget'].items():sections.append('| '+' | '.join([r['dataset'],'/'.join(r['pair']),t,fmt(v)])+' |')
        sections+=['','| Dataset | Gain | Task | Estimate CI |','|---|---|---|---|']
        for ds,gs in data['anneal_gains'].items():
            for name,tasks in gs.items():
                for t,v in tasks.items():sections.append('| '+' | '.join([ds,name,t,fmt(v)])+' |')
        sections+=['','| Model | Task | Parent minus student CI |','|---|---|---|']
        for r in data['drops']:
            for t,v in r['drop'].items():sections.append('| '+' | '.join([r['model'],t,fmt(v)])+' |')
    sections+=['','### Tests, provenance and limitations','',
               'FT14 receipts contain every one-shot NLL check. FT15 records the full-set batch1/batch2 generation comparison and selected batch. FT16 fixes dataset hashes, prompts, task versions, item counts, all five-shot examples and seed before GPU scoring. Raw records and the full analysis remain on node; only this report, compact CSV and proof are delivered.','',
               'TriviaQA one-shot models have likelihood-only scores; no generation was requested, so their EM and jointly-correct TriviaQA cohorts are unavailable.','',confidence['conf_definition'],'',
               'Full per-task drops, slopes, anneal gains, confidence stage probabilities and joint-correct model means: F10b.json and F10c.json on node.','', 'Retained flags:']
    for r in read(root/'flags.json')['flags'] if (root/'flags.json').exists() else cfg['retained_flags']:sections.append('- '+str(r))
    full=locked_report(cfg['previous_report'])+'\n'+'\n'.join(sections)+'\n'
    (root/'REPORT_FULL_v1.11.md').write_text(full)
    with (root/'F10_MODELS.csv').open('x') as f:
        columns=['spec_sha256','model','MMLU_acc_norm','MMLU_acc','MMLU_subject_average','TriviaQA_EM','TriviaQA_F1']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(dict(spec_sha256=SPEC_SHA,model=m,**r) for m,r in newmodels.items())
    write(root/'report_complete.json',dict(complete=(len(newmodels)==42 and sum('TriviaQA_EM' in r for r in newmodels.values())==34 and len(list((root/'nine').glob('*/per_item.json')))==11),raw_new_metric_replay=True,new_models=len(newmodels),confidence_models=len(conf),
                                         old_report_preserved=True,scope='F10a/F10b/F10c',joint_bootstrap_seed=0))
