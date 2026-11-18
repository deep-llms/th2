"""Compact canonical task replay and paired final-checkpoint comparisons."""
import csv,json,math,shutil,tarfile
from pathlib import Path
import numpy as np
from eval.a2_request_bridge import digest
from .data import sha
from .train import write_json

TASKS=('hellaswag','piqa','arc_easy','lambada_openai')
COUNTS=dict(hellaswag=10042,piqa=1838,arc_easy=2376,lambada_openai=5153)
PRIMARY=dict(hellaswag='acc_norm',piqa='acc_norm',arc_easy='acc',lambada_openai='acc')


def compact(raw,prepared,scored,expected_counts=COUNTS):
    if raw['a2_metadata']['prepared_sha256']!=digest(prepared):raise ValueError('Request identity')
    lookup={}
    for r,s in zip(prepared['requests'],scored['scores'],strict=True):
        key=tuple(r['text']);value=(len(r['continuation']),float(s[0]),bool(s[1]))
        lookup.setdefault(key,[]).append(value)
    result={}
    for task in TASKS:
        rows=[]
        for r in sorted(raw['samples'][task],key=lambda r:r['doc_id']):
            responses=r['filtered_resps'];ll=np.array([x[0] for x in responses],dtype=float)
            lengths=[]
            for arg,s in zip(r['arguments'],responses,strict=True):
                found={n for n,v,g in lookup[tuple(arg)] if (v,g)==(float(s[0]),bool(s[1]))}
                if len(found)!=1:raise ValueError('Response changed or token length ambiguous')
                lengths.append(found.pop())
            if not np.isfinite(ll).all() or (ll>0).any():raise ValueError('Invalid log likelihood')
            item=dict(id=r['doc_id'],identity=digest({k:r[k] for k in ('doc','target','arguments')}),ll=ll.tolist(),lengths=lengths)
            if task=='lambada_openai':
                if len(ll)!=1:raise ValueError('LAMBADA request count')
                correct=int(responses[0][1]);item.update(correct=correct,greedy=correct,target_nll=-float(ll[0]))
            else:
                doc=r['doc'];choices=doc['choices'] if task=='hellaswag' else [doc['sol1'],doc['sol2']] if task=='piqa' else doc['choices']['text']
                chars=np.array([len(c) for c in choices]);gold=int(r['target'])
                if (chars<=0).any() or len(chars)!=len(ll) or not 0<=gold<len(ll):raise ValueError('Choice shape')
                acc=int(ll.argmax()==gold);norm=int((ll/chars).argmax()==gold)
                if acc!=r['acc'] or norm!=r['acc_norm']:raise ValueError('Accuracy replay mismatch')
                vals=ll/chars if PRIMARY[task]=='acc_norm' else ll
                correct=norm if PRIMARY[task]=='acc_norm' else acc
                item.update(correct=correct,acc=acc,acc_norm=norm,chars=chars.tolist(),gold=gold,margin=float(vals[gold]-max(v for i,v in enumerate(vals) if i!=gold)))
            if correct!=r[PRIMARY[task]]:raise ValueError('Primary correctness mismatch')
            rows.append(item)
        if len(rows)!=expected_counts[task] or len({x['id'] for x in rows})!=len(rows):raise ValueError('Incomplete item population')
        value=float(np.mean([x['correct'] for x in rows]))
        if abs(value-raw['results'][task][PRIMARY[task]+',none'])>1e-12:raise ValueError('Aggregate differs')
        result[task]=dict(metric=PRIMARY[task],accuracy=value,items=rows)
    return result


def paired(a,b):
    if [(x['id'],x['identity']) for x in a]!=[(x['id'],x['identity']) for x in b]:raise ValueError('Unpaired examples')
    delta=np.array([x['correct']-y['correct'] for x,y in zip(a,b)],dtype=int)
    # Multinomial over the three observed differences is exactly the paired
    # nonparametric item bootstrap, with repeated identical deltas collapsed.
    count=np.array([(delta==k).sum() for k in (-1,0,1)]);rng=np.random.default_rng(901)
    draws=rng.multinomial(len(delta),count/len(delta),size=2000)
    values=(draws[:,2]-draws[:,0])/len(delta)
    return dict(delta=float(delta.mean()),lo=float(np.quantile(values,.025)),hi=float(np.quantile(values,.975)),items=len(delta),wins=int(count[2]),losses=int(count[0]))


def report(base):
    base=Path(base);plan=json.loads((base/'plan.json').read_text());results={};rows=[]
    for e in plan['models']:
        name=e['name'];r=json.loads((base/f'runs/{name}/items.json').read_text());results[name]=r
        for t in TASKS:rows.append(dict(model=name,study=e['study'],arm=e['arm'],seed=e['seed'],task=t,metric=r[t]['metric'],accuracy=r[t]['accuracy'],items=len(r[t]['items'])))
    comparisons={}
    for e in plan['models']:
        baseline=('pilot-D' if e['study']=='pilot' else 'local-C')+f"-s{e['seed']}"
        if baseline==e['name']:continue
        comparisons[e['name']]={t:paired(results[e['name']][t]['items'],results[baseline][t]['items']) for t in TASKS}
    with (base/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    write_json(base/'REPORT.json',dict(rows=rows,comparisons=comparisons,flags=['post_hoc_downstream_screen','two_seeds','unequal_training_compute','short_context_distribution_shift','common_context_truncation','weights_node_only']))
    lines=['# Warm/local-order downstream','','Full zero-shot four-task screen; context512, frozen final checkpoints. Percent accuracy.','','| Model | HellaSwag norm | PIQA norm | ARC-Easy raw | LAMBADA |','|---|---:|---:|---:|---:|']
    for e in plan['models']:lines.append('| '+e['name']+' | '+' | '.join(f"{100*results[e['name']][t]['accuracy']:.3f}" for t in TASKS)+' |')
    lines+=['','Compare R/W against same-seed D; CF/PG against same-seed C. No equal-compute or pooled-training-seed significance claim. REPORT.json contains paired item intervals. Router sees only retained context; candidates share truncation and route. No answer enters routing. Short context (<32) is an explicit transfer distribution shift.']
    (base/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files=[base/n for n in ['REPORT.json','REPORT.md','RESULTS.csv','plan.json','gpu-probe.json','execution/source.json','execution/admission.json']]
    for e in plan['models']:
        files.extend(base/f"runs/{e['name']}/{n}" for n in ('scores.json','items.json','details.json','complete.json','replay-complete.json'))
    with tarfile.open(base/'results.tar.gz','w:gz') as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(base)),recursive=False)
    if (base/'results.tar.gz').stat().st_size>25_000_000:raise ValueError('Small export limit')
    proof=dict(archive_sha256=sha(base/'results.tar.gz'),bytes=(base/'results.tar.gz').stat().st_size,files={str(f.relative_to(base)):sha(f) for f in files},weights_exported=False)
    backup=Path('/mnt/local/_backup/deep-llms_th2/warm_downstream_20261118_a01');backup.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(base/'results.tar.gz',backup/'results.tar.gz')
    if sha(backup/'results.tar.gz')!=proof['archive_sha256']:raise ValueError('Backup mismatch')
    proof['node_backup']=str(backup/'results.tar.gz');write_json(base/'backup.json',proof);write_json(backup/'backup.json',proof)
    return proof
