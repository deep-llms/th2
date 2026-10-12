"""Original-document position probe, deliberately separate from packed PPL."""
from pathlib import Path
import math
import numpy as np
from . import artifacts, protocol, stats
from .run import token_hash


def freeze(rows, tokenizer, output, frozen):
    spec=frozen['position']; selected=[]
    for index,row in enumerate(rows):
        text=row['text']; ids=tokenizer(text,add_special_tokens=False)['input_ids']
        if len(ids)<spec['length']:continue
        ids=ids[:spec['length']]
        selected.append(dict(document_id=f'{index}:{protocol.digest(text)}',token_sha256=protocol.digest(ids),input_ids=ids))
        if len(selected)==spec['documents']:break
    if len(selected)!=spec['documents']:raise ValueError('Not enough qualifying original documents')
    value=dict(protocol_sha256=frozen['protocol_sha256'],tokenizer_sha256=token_hash(tokenizer),
        policy='first qualifying original documents in stored order; tokenize without special tokens; truncate; no appended EOS',
        length=spec['length'],rows=selected)
    value['sample_sha256']=protocol.digest(value);protocol.write_json(output,value)
    return value


def read_sample(path,frozen):
    sample=artifacts.load_json(path)
    body={k:v for k,v in sample.items() if k!='sample_sha256'}
    if protocol.digest(body)!=sample['sample_sha256'] or sample['protocol_sha256']!=frozen['protocol_sha256']:
        raise ValueError('Position sample hash/protocol differs')
    rows=sample['rows']
    if len(rows)!=frozen['position']['documents'] or sample['length']!=frozen['position']['length']:
        raise ValueError('Position sample size differs')
    if len({r['document_id'] for r in rows})!=len(rows):raise ValueError('Duplicate document ID')
    for row in rows:
        ids=row['input_ids']
        if len(ids)!=sample['length'] or any(type(i) is not int or i<0 for i in ids) or protocol.digest(ids)!=row['token_sha256']:
            raise ValueError('Invalid position token sequence/hash')
    return sample


def losses(model,ids):
    """Unreduced FP32 next-token loss, checked against the native training LM sum."""
    import torch
    from torch.nn import functional as F
    from deep_kv.model import Context
    with torch.no_grad():
        logits=model(ids).logits
        loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').reshape(ids.shape[0],-1)
        if not torch.isfinite(loss).all():raise ValueError('Non-finite position loss')
        context=Context(ids,torch.ones_like(ids,dtype=torch.bool),torch.arange(ids.shape[1],device=ids.device).expand_as(ids),torch.zeros_like(ids))
        with torch.autocast(ids.device.type,dtype=model.autocast_dtype,enabled=model.autocast_dtype is not None):
            trained=model.wrapped(context,compute_auxiliary_losses=False)
        delta=abs(float(loss.mean())-float(trained['lm_sum']/trained['lm_count']))
        if delta>1e-4: raise ValueError(f'T8 native training loss disagrees: {delta}')
    return loss.double().cpu().numpy(),delta


def evaluate(checkpoint,sample_path,output,frozen_path,device='cuda',path='production',batch_size=1):
    import torch
    from .run import load_model
    frozen=protocol.read(frozen_path);sample=read_sample(sample_path,frozen)
    output=Path(output)
    if output.exists():raise ValueError('Choose a fresh position output')
    if batch_size<1:raise ValueError('Positive batch size required')
    model,tokenizer,meta=load_model(checkpoint,device,path)
    if token_hash(tokenizer)!=sample['tokenizer_sha256']:raise ValueError('Position tokenizer differs')
    all_losses=np.full((len(sample['rows']),sample['length']),np.nan);errors=[]
    for start in range(0,len(all_losses),batch_size):
        rows=sample['rows'][start:start+batch_size]
        result,error=losses(model,torch.tensor([r['input_ids'] for r in rows],device=device))
        all_losses[start:start+len(rows),1:]=result;errors.append(error)
    output.mkdir(parents=True)
    np.save(output/'losses.npy',all_losses,allow_pickle=False)
    from eval.models import file_hash
    meta.update(status='completed',protocol_sha256=frozen['protocol_sha256'],sample_sha256=sample['sample_sha256'],
        policy=sample['policy'],batch_size=batch_size,T8_max_abs_error=max(errors),
        rows=[{k:r[k] for k in ('document_id','token_sha256')} for r in sample['rows']],losses_sha256=file_hash(output/'losses.npy'))
    protocol.write_json(output/'metadata.json',meta)


def analyze(directories,frozen):
    from eval.models import file_hash
    arrays={};metas={}
    for arm,directory in directories.items():
        root=Path(directory);meta=artifacts.load_json(root/'metadata.json');array=np.load(root/'losses.npy',allow_pickle=False)
        if meta['status']!='completed' or meta['protocol_sha256']!=frozen['protocol_sha256'] or meta['T8_max_abs_error']>1e-4:
            raise ValueError('Invalid position provenance/T8')
        if file_hash(root/'losses.npy')!=meta['losses_sha256']:raise ValueError('Position losses changed')
        if array.shape!=(frozen['position']['documents'],frozen['position']['length']) or not np.isnan(array[:,0]).all() or not np.isfinite(array[:,1:]).all():
            raise ValueError('Invalid position loss array')
        arrays[arm]=array;metas[arm]=meta
    if set(arrays)!={'A10','A20','P20'}:raise ValueError('Position probe requires both pairs')
    for key in ('rows','policy','sample_sha256','tokenizer_sha256'):
        if any(m[key]!=metas['A20'][key] for m in metas.values()):raise ValueError('Position document/token identities differ')
    B=frozen['resamples'];seed=frozen['analysis_seed'];out=dict(provenance=metas,pairs={},difference_of_profiles={})
    for name,a,b in [('P20-A20','A20','P20'),('A20-A10','A10','A20')]:
        reduction=arrays[a]-arrays[b];all_s=reduction[:,1:].sum(1);all_c=np.full(len(all_s),reduction.shape[1]-1)
        overall=stats.ratio(all_s,all_c,B,seed);buckets={}
        for lo,hi in frozen['position']['buckets']:
            s=reduction[:,lo:hi].sum(1);count=reduction[:,lo:hi].shape[1]
            if not count:buckets[f'{lo}:{hi}']=None;continue
            # All selected documents have equal length. Joint ratio of means
            # equals ratio of token-averaged reductions, preserving covariance.
            normalized=stats.progress_ratio([(s/count,all_s/all_c)],B,seed)
            buckets[f'{lo}:{hi}']=dict(reduction=stats.ratio(s,np.full(len(s),count),B,seed),normalized_profile=normalized)
        out['pairs'][name]=dict(overall=overall,buckets=buckets)
    # Direct profile difference uses the same resampled documents in BOTH pairs.
    rng=np.random.default_rng(seed);N=len(arrays['A20']);pairs=[arrays['A20']-arrays['P20'],arrays['A10']-arrays['A20']]
    for lo,hi in frozen['position']['buckets']:
        key=f'{lo}:{hi}'
        if not all(out['pairs'][p]['buckets'][key] is not None and out['pairs'][p]['buckets'][key]['normalized_profile']['estimable'] for p in out['pairs']):
            out['difference_of_profiles'][key]=None;continue
        sums=[(d[:,lo:hi].mean(1),d[:,1:].mean(1)) for d in pairs];boot=np.empty(B)
        for start in range(0,B,32):
            k=min(32,B-start);idx=rng.integers(N,size=(k,N))
            with np.errstate(divide='ignore',invalid='ignore'):
                boot[start:start+k]=sums[0][0][idx].mean(1)/sums[0][1][idx].mean(1)-sums[1][0][idx].mean(1)/sums[1][1][idx].mean(1)
        out['difference_of_profiles'][key]=dict(estimate=float(sums[0][0].mean()/sums[0][1].mean()-sums[1][0].mean()/sums[1][1].mean()),ci=stats.interval(boot))
    return out
