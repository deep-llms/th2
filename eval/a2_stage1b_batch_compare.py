"""Full P300 batch1/batch2 decision and aligned diagnostics, without GPU work."""
import math
from eval.downstream_suite import COUNTS, PRIMARY
from eval.a2_request_bridge import digest


def padding_pairs(requests,max_length=4096):
    lengths=[min(len(r['context'] or [0])+len(r['continuation']),max_length+1)-1 for r in requests]
    order=sorted(range(len(lengths)),key=lambda i:-lengths[i]);out=[None]*len(order)
    for start in range(0,len(order),2):
        pair=order[start:start+2];width=max(lengths[i] for i in pair)
        padded=any(lengths[i]<width for i in pair)
        for i in pair:out[i]=dict(pair_indices=pair,pair_has_padding=padded,request_padding_tokens=width-lengths[i])
    return out


def ll_differences(prepared,batch1,batch2):
    for key in ('prepared_sha256','request_sha256','checkpoint_identity_sha256',
                'likelihood_requests','precision','max_length','packages'):
        if batch1[key]!=batch2[key]:raise ValueError('Unaligned batch scores: '+key)
    if (batch1['batch_size']!=1 or batch2['batch_size']!=2
            or batch1['prepared_sha256']!=digest(prepared)
            or batch1['request_sha256']!=digest(prepared['requests'])):
        raise ValueError('Wrong full batch comparison inputs')
    requests=prepared['requests'];n=len(requests)
    if n!=78259 or any(len(x['scores'])!=n or x['likelihood_requests']!=n for x in (batch1,batch2)):
        raise ValueError('Full 78259 request comparison required')
    padding=padding_pairs(requests,prepared['protocol']['max_length']);rows=[]
    for i,(a,b,r,p) in enumerate(zip(batch1['scores'],batch2['scores'],requests,padding)):
        if not all(math.isfinite(v) for v in (a[0],b[0])):raise ValueError('Nonfinite LL')
        delta=b[0]-a[0]
        rows.append(dict(index=i,request_sha256=digest(r),difference=delta,
            abs_difference=abs(delta),nats_per_token=delta/len(r['continuation']),
            greedy_exact_match_flipped=a[1]!=b[1],**p))
    def stats(subset):
        return dict(requests=len(subset),nonzero=sum(x['abs_difference']!=0 for x in subset),
            max_abs=max((x['abs_difference'] for x in subset),default=0),
            mean_abs=sum(x['abs_difference'] for x in subset)/len(subset) if subset else None,
            mean_signed=sum(x['difference'] for x in subset)/len(subset) if subset else None)
    largest=max(rows,key=lambda x:x['abs_difference'])
    return dict(summary=stats(rows),largest=largest,
        padded_pairs=stats([r for r in rows if r['pair_has_padding']]),
        unpadded_pairs=stats([r for r in rows if not r['pair_has_padding']]),
        differences_only_in_padded_pairs=all(r['pair_has_padding'] for r in rows if r['abs_difference']),
        rows=rows)


def decide_batch(task_accuracy1,task_accuracy2):
    if set(task_accuracy1)!=set(COUNTS) or set(task_accuracy2)!=set(COUNTS):
        raise ValueError('All nine task accuracies required')
    if not all(math.isfinite(x) and 0<=x<=1 for d in (task_accuracy1,task_accuracy2) for x in d.values()):
        raise ValueError('Invalid accuracy')
    differences={t:task_accuracy2[t]-task_accuracy1[t] for t in COUNTS}
    mc=sum(d for t,d in differences.items() if t!='lambada_openai')/8
    lambada=differences['lambada_openai']
    # Roundoff tolerance only at the inclusive 0.1 percentage-point boundary.
    passed=abs(mc)<=.001+1e-15 and abs(lambada)<=.001+1e-15
    return dict(batch2_accepted=passed,production_batch_size=2 if passed else 1,
        mc_mean_difference_pp=100*mc,lambada_difference_pp=100*lambada,
        limit_pp=.1,per_task={t:dict(batch1=task_accuracy1[t],batch2=task_accuracy2[t],
            difference_pp=100*differences[t],metric=PRIMARY[t]) for t in COUNTS})


def prediction_flips(samples1,samples2,choices_for_doc,*,normalized=False,lambada=False):
    """MC predictions use the pinned harness's own doc_to_choice callable.

    Report exact-match flips for LAMBADA separately: the canonical scorer stores
    correctness, not generated word strings, so this cannot assert two incorrect
    generated words were the same. That limitation must appear in the report.
    """
    a={r['doc_id']:r for r in samples1};b={r['doc_id']:r for r in samples2}
    if len(a)!=len(samples1) or len(b)!=len(samples2) or set(a)!=set(b):raise ValueError('Unpaired samples')
    flips=[]
    for i in sorted(a):
        x,y=a[i],b[i]
        if any(x[k]!=y[k] for k in ('doc','arguments','target')):raise ValueError('Changed prompts')
        if lambada:
            px,py=bool(x['filtered_resps'][0][1]),bool(y['filtered_resps'][0][1])
        else:
            choices=choices_for_doc(x['doc']);lengths=[float(len(c)) for c in choices]
            scores=[]
            for row in (x,y):
                vals=[float(r[0]) for r in row['filtered_resps']]
                if len(vals)!=len(lengths) or any(n<=0 for n in lengths):raise ValueError('Choice alignment changed')
                if normalized:vals=[v/n for v,n in zip(vals,lengths)]
                scores.append(max(range(len(vals)),key=lambda j:vals[j]))
            px,py=scores
        if px!=py:flips.append(i)
    return dict(items=len(a),flips=len(flips),flipped_doc_ids=flips,
                definition='LAMBADA exact-match correctness' if lambada else 'MC primary-metric choice argmax')


def lambada_token_prediction_flips(samples, prepared, predictions1, predictions2):
    """Count changed teacher-forced predicted continuations, including both-wrong.

    Canonical LAMBADA has one request per sample. Duplicate text requests must
    have consistent captured predictions, otherwise pairing is ambiguous and
    the diagnostic is rejected instead of silently choosing one occurrence.
    """
    for key in ('prepared_sha256','request_sha256','checkpoint_identity_sha256'):
        if predictions1[key]!=predictions2[key]:raise ValueError('Unaligned argmax receipts')
    if (predictions1['batch_size']!=1 or predictions2['batch_size']!=2
            or predictions1['prepared_sha256']!=digest(prepared)
            or any(len(x['predictions'])!=len(prepared['requests']) for x in (predictions1,predictions2))):
        raise ValueError('Incomplete argmax receipts')
    indices={}
    for i,r in enumerate(prepared['requests']):indices.setdefault(tuple(r['text']),[]).append(i)
    flipped=[];paired_cache={}
    for sample in samples:
        if len(sample['arguments'])!=1:raise ValueError('Expected one LAMBADA request')
        matches=indices.get(tuple(sample['arguments'][0]),[])
        if not matches:raise ValueError('Missing canonical LAMBADA request')
        key=tuple(sample['arguments'][0])
        if key not in paired_cache:
            pairs={(tuple(predictions1['predictions'][i]),tuple(predictions2['predictions'][i])) for i in matches}
            if len(pairs)!=1:raise ValueError('Ambiguous duplicate LAMBADA prediction pairing')
            paired_cache[key]=pairs.pop()
        a,b=paired_cache[key]
        if a!=b:flipped.append(sample['doc_id'])
    return dict(items=len(samples),flips=len(flipped),flipped_doc_ids=flipped,
                definition='teacher-forced continuation argmax tokens, including two incorrect predictions')
