"""Frozen-backbone STOP head, independent calibration, held-out PPL and latency.

Labels are fixed-depth next-token-loss heuristics on TRAINING rows only. They
are not oracle mixed-exit labels. The deployed mixed policy is scored directly.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from bdot.model import BDOTForCausalLM
from packed_training import PackedCollator, model_inputs
from train import DataArguments, prepare_validation_data
from training_data import prepare_training_data, require_resume_data


def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')


def choose_policy(ctl, policy):
    ctl.policy=policy['policy']
    for k in ('fixed_depth','constant_prob','threshold'):
        if k in policy: setattr(ctl,k,policy[k])


def batches(dataset, backend, device, size=4):
    collate=PackedCollator(attention_backend=backend)
    for start in range(0,len(dataset),size):
        rows=[dataset[i] for i in range(start,min(start+size,len(dataset)))]
        data=collate(rows)
        yield model_inputs({k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in data.items()})


def token_losses(logits, labels):
    # Chunking bounds FP32 vocabulary-sized temporary allocations.
    logits=logits[:,:-1].reshape(-1,logits.shape[-1]);labels=labels[:,1:].reshape(-1)
    return torch.cat([F.cross_entropy(x.float(),y,ignore_index=-100,reduction='none')
                      for x,y in zip(logits.split(1024),labels.split(1024))])


def score(model, dataset, backend, policy, size=4):
    choose_policy(model.stop_controller,policy);model.eval_counter=0
    total=count=depths=inputs=0
    started=time.perf_counter()
    with torch.inference_mode(),torch.autocast(model.device.type,dtype=torch.bfloat16,enabled=model.device.type=='cuda'):
        for data in batches(dataset,backend,model.device,size):
            labels=data.pop('labels');out=model(**data)
            total+=float(token_losses(out.logits,labels).double().sum())
            count+=int(labels[:,1:].ne(-100).sum())
            depths+=int(model.stop_controller.last_depths.sum());inputs+=labels.numel()
    if not count or not math.isfinite(total): raise ValueError('Invalid STOP evaluation')
    return dict(policy=policy,loss=total/count,perplexity=math.exp(total/count),
        num_tokens=count,num_sequences=len(dataset),mean_middle_steps=depths/inputs,
        seconds=time.perf_counter()-started)


def fit_head(model, dataset, backend, size=4, samples_per_row=32, epochs=20):
    ctl=model.stop_controller;S=model.options.S
    features=[];targets=[];steps=[]
    sample_rng=torch.Generator().manual_seed(371)
    choose_policy(ctl,dict(policy='none'));model.eval()
    with torch.inference_mode(),torch.autocast(model.device.type,dtype=torch.bfloat16,enabled=model.device.type=='cuda'):
        for data in batches(dataset,backend,model.device,size):
            labels=data.pop('labels');length=labels.shape[1]-1
            ctl.capture_states=True
            full=token_losses(model(**data).logits,labels)
            states=ctl.last_states
            ctl.capture_states=False;ctl.last_states=[]
            # Uniform positions among valid targets; fixed across both trunks.
            valid=labels[:,1:].ne(-100).cpu()
            chosen=[]
            for row,mask in enumerate(valid):
                candidates=mask.nonzero().flatten()
                picks=candidates[torch.randperm(len(candidates),generator=sample_rng)[:samples_per_row]]
                chosen.append(picks+row*length)
            chosen=torch.cat(chosen).to(model.device)
            for depth in (ctl.options.exit_steps or range(ctl.options.min_steps,S)):
                choose_policy(ctl,dict(policy='fixed',fixed_depth=depth))
                early=token_losses(model(**data).logits,labels)
                features.append(states[depth-1][:,:-1].reshape(-1,model.config.hidden_size)[chosen].float().cpu())
                targets.append(((early[chosen]-full[chosen])<=.01).float().cpu())
                steps.append(torch.full((len(chosen),),depth-1,dtype=torch.long))
            choose_policy(ctl,dict(policy='none'))
    x,y,s=map(torch.cat,(features,targets,steps))
    # Clone outside inference_mode: these tensors participate in head autograd.
    x=x.clone().to(model.device);y=y.clone().to(model.device);s=s.clone().to(model.device)
    model.requires_grad_(False)
    ctl.head.requires_grad_(True);ctl.bias.requires_grad_(True)
    opt=torch.optim.AdamW([ctl.head.weight,ctl.bias],lr=1e-3,weight_decay=0.)
    generator=torch.Generator(device=model.device).manual_seed(812)
    losses=[]
    for epoch in range(epochs):
        order=torch.randperm(len(y),generator=generator,device=model.device)
        total=0.
        for idx in order.split(2048):
            opt.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(ctl.logits(x[idx],s[idx]),y[idx])
            loss.backward();opt.step();total+=float(loss.detach())*len(idx)
        losses.append(total/len(y))
    ctl.requires_grad_(False);ctl.policy='none'
    return dict(examples=len(y),positive_fraction=float(y.mean()),bce_by_epoch=losses,
        label='fixed-depth NLL minus full-depth NLL <= 0.01 nats on valid next-token targets',
        sampling_seed=371,head_shuffle_seed=812,epochs=epochs,lr=.001,
        caveat='Full-depth features/fixed-depth labels are heuristic; mixed exits are evaluated directly.')


def random_probability(mean_depth, steps=8, minimum=2):
    lo,hi=0.,1.
    for _ in range(50):
        mid=(lo+hi)/2
        expected=minimum+sum((1-mid)**i for i in range(1,steps-minimum+1))
        if expected>mean_depth:lo=mid
        else:hi=mid
    return (lo+hi)/2


def latency(model, policy, tokens, batch_size, context=128, repeats=32):
    choose_policy(model.stop_controller,policy)
    if len(tokens)<batch_size: raise ValueError('Need distinct timing sequences for the requested batch')
    ids=torch.tensor([row[:context+repeats+4] for row in tokens[:batch_size]],device=model.device)
    if ids.shape[1]!=context+repeats+4: raise ValueError('Latency token fixture too short')
    trials=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(3):
            model.eval_counter=0
            torch.cuda.synchronize();start=time.perf_counter()
            result=model(input_ids=ids[:,:context],use_cache=True,logits_to_keep=1)
            torch.cuda.synchronize();prefill=time.perf_counter()-start
            cache=result.past_key_values
            for i in range(context,context+4):
                cache=model(input_ids=ids[:,i:i+1],past_key_values=cache,use_cache=True).past_key_values
            torch.cuda.synchronize();start=time.perf_counter()
            depth=0
            for i in range(context+4,ids.shape[1]):
                cache=model(input_ids=ids[:,i:i+1],past_key_values=cache,use_cache=True).past_key_values
                # Keep counters on GPU until timed work is over.
                depth=depth+model.stop_controller.last_depths.sum()
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            trials.append(dict(prefill_seconds=prefill,decode_ms=1000*elapsed/repeats,
                               mean_middle_steps=float(depth)/(repeats*batch_size)))
    return dict(policy=policy,batch=batch_size,context=context,trials=trials,
                decode_ms=statistics.median(x['decode_ms'] for x in trials))


def long_fragments(dataset, minimum=548, count=32):
    found=[]
    for row in dataset:
        segments=row['segments'];start=0
        for end in range(1,len(segments)+1):
            if end==len(segments) or segments[end]!=segments[start]:
                if end-start>=minimum:
                    found.append(row['input_ids'][start:end])
                    if len(found)==count:return found
                start=end
    raise ValueError('Insufficient long single-document fragments for cached timing')


def cached_policy_check(model,policy,tokens):
    choose_policy(model.stop_controller,policy)
    ids=torch.tensor([tokens[:128]],device=model.device)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        full=model(input_ids=ids,use_cache=False).logits
        full_depth=model.stop_controller.last_depths.clone()
        cache=None;outputs=[];depths=[]
        for i in range(ids.shape[1]):
            result=model(input_ids=ids[:,i:i+1],past_key_values=cache,use_cache=True)
            cache=result.past_key_values;outputs.append(result.logits)
            depths.append(model.stop_controller.last_depths)
        recurrent=torch.cat(outputs,1);depths=torch.cat(depths,1)
        rms=float((full-recurrent).float().square().mean().sqrt())
        mismatch=float(depths.ne(full_depth).float().mean())
        loss_gap=float((token_losses(recurrent,ids)-token_losses(full,ids)).mean())
    # Report BF16 decision discontinuities; do not silently infer cached quality
    # from parallel PPL if mixed decisions differ substantially.
    return dict(logit_rms=rms,depth_mismatch_fraction=mismatch,nll_gap=loss_gap,
                passed=rms<.08 and mismatch<=.01 and abs(loss_gap)<.01)


def run(checkpoint,parent_recipe,output):
    from packed_fa4 import register_backend
    backend=register_backend();torch.set_num_threads(1)
    recipe=read(parent_recipe)
    tokenizer=AutoTokenizer.from_pretrained(recipe['tokenizer_name'],local_files_only=True)
    fields=DataArguments.__dataclass_fields__
    data_args=DataArguments(**{k:v for k,v in recipe.items() if k in fields})
    settings=SimpleNamespace(data_seed=0,seed=0,main_process_first=lambda **kw:nullcontext())
    train,identity=prepare_training_data(data_args,settings,tokenizer,recipe['tokenizer_name'],1024,'fa4')
    parent=Path(recipe['output_dir'])/'checkpoint-10000';require_resume_data(parent,identity)
    # Parent 10k + four equal adaptation branches' next 1k; fresh disjoint slices.
    start=11000*512
    if start+512>len(train):raise ValueError('Insufficient unused training rows for head/calibration')
    fit=train.select(range(start,start+256));cal=train.select(range(start+256,start+512))
    model=BDOTForCausalLM.from_pretrained(checkpoint,attn_implementation=backend).cuda().eval()
    if int(model.completed_steps)!=1000:raise ValueError('Adaptation checkpoint incomplete')
    output.mkdir(parents=True,exist_ok=False)
    head=fit_head(model,fit,'fa4')
    torch.save(model.stop_controller.state_dict(),output/'stop_head.pt')
    write(output/'head.json',dict(status='ok',training_rows=[start,start+256],
        calibration_rows=[start+256,start+512],**head))
    policies=[dict(policy='none'),*[dict(policy='fixed',fixed_depth=k) for k in (2,4,6)],
              *[dict(policy='learned',threshold=t) for t in (.25,.5,.7,.85,.95,.99,.999)]]
    calibration=[score(model,cal,'fa4',p) for p in policies]
    full=calibration[0]
    feasible=[r for r in calibration if r['policy']['policy']=='learned'
              and r['loss']<=full['loss']+.01]
    chosen=min(feasible,key=lambda r:r['mean_middle_steps']) if feasible else full
    matched=dict(policy='constant',constant_prob=random_probability(chosen['mean_middle_steps']))
    calibration.append(score(model,cal,'fa4',matched))
    write(output/'calibration.json',dict(status='ok',results=calibration,chosen=chosen,matched_random=matched))
    # A single threshold is frozen before final validation. Never tune on it.
    from dataclasses import replace
    validation,_=prepare_validation_data(replace(data_args,eval_max_sequences=None),settings,
        tokenizer,recipe['tokenizer_name'],1024,'fa4',identity)
    final=[score(model,validation,'fa4',dict(policy='none'))]
    if (final[0]['num_tokens'],final[0]['num_sequences'])!=(9915698,9704):
        raise ValueError('Final validation size changed')
    expected=read(Path(checkpoint).parent/'final_eval.json')['eval_loss']
    if abs(final[0]['loss']-expected)>1e-4:
        raise ValueError('Full-depth carry changed beyond minibatch rounding tolerance')
    if chosen['policy']['policy']!='none':
        final.append(score(model,validation,'fa4',chosen['policy']))
        final.append(score(model,validation,'fa4',matched))
    # Cached SDPA timing is shared across all controls, incl router/head/KV/head.
    model.set_attn_implementation('sdpa')
    tokens=long_fragments(cal)
    cached_checks={label:cached_policy_check(model,p,tokens[0]) for label,p in
                   [('full',dict(policy='none')),('chosen',chosen['policy'])]}
    measured=[latency(model,p,tokens,b) for p in [dict(policy='none'),chosen['policy'],matched]
              for b in (1,32)]
    measured += [latency(model,p,tokens,b,context=512) for p in [dict(policy='none'),chosen['policy']]
                 for b in (1,32)]
    write(output/'result.json',dict(status='ok',checkpoint=str(checkpoint),head=head,
        chosen=chosen,final=final,latency=measured,calibration=calibration,
        checkpoint_step=1000,parent_step=10000,
        cached_checks=cached_checks,
        timing_note='SDPA cached whole-model teacher-forced decode; same token stream; FA4 packed PPL.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','parent-recipe','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.checkpoint,a.parent_recipe,a.output)


if __name__=='__main__':main()
