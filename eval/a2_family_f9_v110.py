"""Closed v1.10 inventory and secondary continuous metrics; CPU only."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from eval.a2_family_f9 import read,items
from eval.a2_request_bridge import digest
from eval.downstream_suite import COUNTS,PRIMARY
from scripts.a2_quant_io import write as atomic_write

SPEC_SHA='1cf76d4e7a9d321e01f58ef704fc722ad186f2e966b3b34280141fe183b158cf'
EVALUATOR_COMMIT='2a74421e4dc6f4f6669af1769e1ae07828b031c2'

def write(path,value):return atomic_write(path,dict(value,spec_sha256=SPEC_SHA))

def select(entries):
    groups=[[],[],[],[],[]];excluded=[]
    for e in entries:
        g=None
        if e['role']!='student':
            if e['family']=='smollm3' or e['dataset'] in ('dclm','fwe'):g=0
        elif e['seed']==2030 and e['budget']>=.2:
            if e['family']=='opensci' and e['cut']=='L6' and e['variant']=='drop' and e['lr']==.001:
                if e['part'] in ('Stage1','DepthC') and e['dataset']=='dclm' and e['arm'] in ('P300','P1T','X72'):g=1
                elif e['part']=='Stage1b' and e['dataset']=='fwe' and e['arm'] in ('P300','P1T','X72'):g=2
                elif e['part'] in ('Breadth','F6','F7') and e['dataset'] in ('dclm','fwe') and e['arm'] in ('S72','S188'):g=3
            elif e['family']=='smollm3' and e['part'] in ('F2','F3','F4') and e['dataset']=='fwe' and e['lr']==.0001:
                g=4
        if g is None:excluded.append(e['model_id'])
        else:groups[g].append(e)
    for group in groups:group.sort(key=lambda e:(e.get('budget') or 0,e['dataset'],e['arm'],e['model_id']))
    keys=[(e['family'],e['role'],e['dataset'],e['arm'],e['budget']) for group in groups for e in group]
    if len(keys)!=len(set(keys)):raise ValueError('Multiple reported endpoints; resolve NLL lineage')
    return groups,excluded

def choices_and_gold(task,row):
    doc=row['doc']
    if task=='hellaswag':choices=doc['choices']
    elif task in ('arc_easy','arc_challenge','openbookqa'):choices=doc['choices']['text']
    elif task=='piqa':choices=[doc['sol1'],doc['sol2']]
    elif task=='boolq':choices=['no','yes']
    elif task=='sciq':choices=[doc[k] for k in ('distractor1','distractor2','distractor3','correct_answer')]
    elif task=='winogrande':
        prefix=doc['sentence'][:doc['sentence'].index('_')]
        return [prefix+doc['option1'],prefix+doc['option2']],int(doc['answer'])-1
    else:raise ValueError('Unregistered task')
    return choices,int(row['target'])

def export_items(replay,prepared,scored):
    """Recover exact choice scores/lengths and prove they reproduce accuracy.

    lm-eval 0.4.13 acc_norm divides by character length of doc_to_choice,
    excluding the target delimiter. LAMBADA secondary NLL is -target LL per
    item, not divided by its token count (which is recorded separately).
    """
    from eval.a2_requests import validate_scored
    validate_scored(prepared,scored)
    if replay['a2_metadata']['prepared_sha256']!=digest(prepared):raise ValueError('Replay request identity differs')
    if replay['a2_metadata']['protocol']!=prepared['protocol']:raise ValueError('Replay protocol differs')
    reference=items(replay);lookup=defaultdict(list)
    for req,score in zip(prepared['requests'],scored['scores'],strict=True):
        lookup[tuple(req['text'])].append((len(req['continuation']),float(score[0]),bool(score[1])))
    exported={};records={}
    for task in COUNTS:
        values=[];rows=[]
        for row in sorted(replay['samples'][task],key=lambda r:r['doc_id']):
            ll=np.array([x[0] for x in row['filtered_resps']],dtype=np.float64)
            lengths=[]
            for arg,resp in zip(row['arguments'],row['filtered_resps'],strict=True):
                found=[n for n,v,g in lookup[tuple(arg)] if v==float(resp[0]) and g==bool(resp[1])]
                if not found or len(set(found))!=1:raise ValueError('Raw request/response or token length mismatch')
                lengths.append(found[0])
            if not np.isfinite(ll).all() or min(lengths)<=0:raise ValueError('Invalid choice likelihood/length')
            out=dict(doc_id=row['doc_id'],raw_loglikelihoods=ll.tolist(),choice_token_lengths=lengths,
                correctness=row[PRIMARY[task].split(',')[0]])
            if task=='lambada_openai':
                if len(ll)!=1:raise ValueError('LAMBADA requires one target')
                out.update(target_NLL=float(-ll[0]),target_token_count=lengths[0],accuracy_scores=ll.tolist())
                values.append(float(ll[0]))  # utility = -NLL, so positive drop means worse
            else:
                choices,gold=choices_and_gold(task,row);chars=np.array([len(c) for c in choices],dtype=float)
                if len(chars)!=len(ll) or min(chars)<=0 or not 0<=gold<len(ll):raise ValueError('Invalid canonical choices')
                metric=PRIMARY[task].split(',')[0]
                s=ll/chars if metric=='acc_norm' else ll
                if int(np.argmax(s)==gold)!=out['correctness']:raise ValueError('Reconstructed task accuracy mismatch')
                q=np.exp(s-s.max());pc=float(q[gold]/q.sum());values.append(pc)
                out.update(choice_character_lengths=chars.astype(int).tolist(),accuracy_scores=s.tolist(),
                    gold_choice=gold,p_correct=pc,normalization='choice_character_length' if metric=='acc_norm' else 'raw')
            rows.append(out)
        exported[task]=rows
        records[task]=dict(identity=reference[task]['identity'],values=values)
    return dict(tasks=exported,continuous_records=records,
                LAMBADA_definition='minus total target loglikelihood per item; unweighted item mean'),reference

def sampler(records,resamples=10000,seed=0):
    """One joint item bootstrap reused across every model and contrast."""
    names=list(records);rng=np.random.default_rng(seed);draws={}
    for task in sorted(COUNTS):
        rows=[records[n][task] for n in names]
        if len({r['identity'] for r in rows})!=1:raise ValueError('Unpaired evaluation items')
        matrix=np.asarray([r['values'] for r in rows],dtype=np.float64).T;n=len(matrix)
        if n!=COUNTS[task] or not np.isfinite(matrix).all():raise ValueError('Invalid item coverage')
        blocks=[]
        for start in range(0,resamples,64):
            weights=rng.multinomial(n,np.full(n,1/n),size=min(64,resamples-start))
            blocks.append(weights@matrix/n)
        draws[task]=(matrix.mean(0),np.concatenate(blocks))
    for label,ts in [('MC8',[t for t in COUNTS if t!='lambada_openai']),('MC7',[t for t in COUNTS if t not in ('lambada_openai','boolq')])]:
        draws[label]=(sum(draws[t][0] for t in ts)/len(ts),sum(draws[t][1] for t in ts)/len(ts))
    def subset(selected):
        ix=[names.index(n) for n in selected]
        return list(selected),{t:(p[ix],d[:,ix]) for t,(p,d) in draws.items()}
    return subset

def continuous_report(entries,replays,records):
    from eval.a2_family_f9_report import analyze
    r=analyze(entries,replays,item_records=records,draw_sampler=sampler(records))
    # Existing readout treats all utility contrasts as percentage points.
    # Convert LAMBADA contrasts back to nats; kept-share ratios stay unitless.
    def convert(obj,ratio=False):
        if isinstance(obj,dict):
            for k,v in obj.items():
                isratio=ratio or 'kept_share' in k or 'over_delta_g' in k
                if k=='lambada_openai' and not isratio:
                    if isinstance(v,dict) and 'estimate' in v:
                        v['estimate']/=100;v['ci95']=[x/100 for x in v['ci95']]
                    elif isinstance(v,(int,float)) and obj is not r.get('models'):obj[k]=v/100
                else:convert(v,isratio)
        elif isinstance(obj,list):
            for v in obj:convert(v,ratio)
    # Restrict conversion to contrasts (model means are already raw utility).
    for key in ('lead','pooled','A_init_down','drops','reference','budget_trends','anneal_gains'):convert(r[key])
    for row in r['models'].values():
        row['LAMBADA_target_NLL']=-row.pop('lambada_openai');row.pop('LAMBADA_perplexity',None)
    r['units']='MC p_correct contrasts percentage points; LAMBADA target NLL contrasts nats/item; kept shares fractions'
    r['secondary']=True
    return r


def render_secondary(accuracy,continuous):
    def val(v):
        if v is None:return 'N/A'
        return f"{v['estimate']:.6f} [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]"
    lines=['### Continuous metric (secondary), alongside accuracy','',
        'MC uses softmax of the exact accuracy scores; acc_norm uses character lengths. LAMBADA uses total target NLL per item, without token division. Positive penalties mean worse students. Intervals: joint item bootstrap, 10,000 draws, seed 0.','',
        '| Dataset | Pair | Budget | Metric | Accuracy E (pp, CI95) | Continuous E (pp or nats, CI95) | Accuracy kept | Continuous kept | Primary / secondary reading |',
        '|---|---|---|---|---|---|---|---|---|']
    def key(r):return (r['dataset'],tuple(r['pair']),r['budget'],r['cut'],r['lr'],r['seed'])
    am={key(r):r for r in accuracy['pooled']+accuracy['lead']}
    for r in continuous['pooled']+continuous['lead']:
        a=am[key(r)]
        for t in ('MC8','MC7','lambada_openai'):
            lines.append('| '+' | '.join(map(str,[r['dataset'],'/'.join(r['pair']),r['budget'],t,
                val(a['E_down'][t]),val(r['E_down'][t]),val(a['accuracy_kept_share'][t]),
                val(r['accuracy_kept_share'][t]),a['reading'][t]+' / '+r['reading'][t]]))+' |')
    lines+=['','### Continuous X72 contrast beside accuracy','',
        '| Dataset | Budget | Metric | Accuracy A_init (pp, CI95) | Continuous A_init (pp or nats, CI95) |',
        '|---|---|---|---|---|']
    for a,r in zip(accuracy['A_init_down'],continuous['A_init_down'],strict=True):
        if a['models']!=r['models']:raise ValueError('Contrast rows not aligned')
        for t in ('MC8','MC7','lambada_openai'):
            lines.append('| '+' | '.join(map(str,[r['dataset'],r['budget'],t,val(a['A_init_down'][t]),val(r['A_init_down'][t])]))+' |')
    lines+=['','Per-model continuous task means and all per-task contrasts: continuous.json. Raw and normalized choice scores, lengths and correctness: per_item/.']
    return '\n'.join(lines)+'\n'
