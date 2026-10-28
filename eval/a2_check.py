"""Representative native-model likelihood check before A2 downstream execution.

Checks the exact scoring path against independent per-request cross-entropy and
compares batch sizes. No training or GPU ownership operations. Run under the
existing supervisor only after the previous workload's terminal handoff.
"""
import argparse
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F

os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='offline')
from efficient_llm.a2.model import autocast
from eval.a2_checkpoint import load_verified,verify_identity
from eval.a2_likelihood import score_requests
from eval.a2_request_bridge import digest
from scripts.a2_remaining import read_locked


def select_requests(prepared):
    if prepared['request_sha256']!=digest(prepared['requests']):raise ValueError('Changed requests')
    rows=prepared['requests']
    if not rows:raise ValueError('No requests')
    # Fixed selection, independent of answers/model scores: first examples and
    # extremes of input and continuation lengths, including truncated inputs.
    chosen=list(range(min(4,len(rows))))
    for key in (lambda r:len(r['context'])+len(r['continuation']),lambda r:len(r['continuation'])):
        ordered=sorted(range(len(rows)),key=lambda i:key(rows[i]))
        chosen+=ordered[:2]+ordered[-2:]
    indices=list(dict.fromkeys(chosen))
    return indices,[rows[i] for i in indices]


@torch.inference_mode()
def check_model(model,requests,*,max_length):
    single=score_requests(model,requests,batch_size=1,max_length=max_length)
    batch=score_requests(model,requests,batch_size=2,max_length=max_length)
    manual=[]
    for row in requests:
        context=row['context'] or [0];target=row['continuation']
        # Independent full joined-sequence labels; discard the initial token,
        # then mask all retained context labels with cross_entropy ignore_index.
        joined=(context+target)[-(max_length+1):]
        ids=torch.tensor([joined[:-1]],device=model.device)
        labels=torch.tensor(joined[1:],device=model.device)
        labels[:-len(target)]=-100
        with autocast(model.device):
            logits=model(input_ids=ids,use_cache=False,
                position_ids=torch.arange(ids.shape[1],device=model.device)[None]).logits[0]
        value=-F.cross_entropy(logits.float(),labels,ignore_index=-100,reduction='sum')
        greedy=bool(torch.equal(logits[-len(target):].argmax(-1),torch.tensor(target,device=model.device)))
        manual.append((float(value),greedy))
    lengths=[len(row['continuation']) for row in requests]
    reference_errors=[abs(a[0]-b[0])/n for a,b,n in zip(single['scores'],manual,lengths)]
    batch_errors=[abs(a[0]-b[0])/n for a,b,n in zip(single['scores'],batch['scores'],lengths)]
    greedy_reference=sum(a[1]!=b[1] for a,b in zip(single['scores'],manual))
    greedy_batch=sum(a[1]!=b[1] for a,b in zip(single['scores'],batch['scores']))
    # Same-input independent algebra is strict. BF16 batching can select
    # different GEMM kernels; report measured differences, do not call bitwise.
    passed=(max(reference_errors)<=2e-5 and max(batch_errors)<=.01
            and greedy_reference==0 and greedy_batch==0
            and single['truncated_requests']==batch['truncated_requests'])
    return dict(passed=passed,requests=len(requests),max_length=max_length,
        manual_max_nats_per_token=max(reference_errors),batch_max_nats_per_token=max(batch_errors),
        greedy_reference_mismatches=greedy_reference,greedy_batch_mismatches=greedy_batch,
        truncated_requests=single['truncated_requests'],
        thresholds=dict(manual_nats_per_token=2e-5,batch_nats_per_token=.01,greedy_mismatches=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('identity','identity-sha256','prepared','prepared-sha256','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();out=Path(a.output)
    if out.exists():p.error('Output exists')
    identity=read_locked(dict(path=a.identity,sha256=a.identity_sha256))
    prepared=read_locked(dict(path=a.prepared,sha256=a.prepared_sha256))
    indices,requests=select_requests(prepared)
    model=load_verified(identity,a.device)
    check=check_model(model,requests,max_length=prepared['protocol']['max_length'])
    verify_identity(identity)
    result=dict(check,checkpoint_identity_sha256=digest(identity),prepared_sha256=digest(prepared),
                selected_request_indices=indices,precision='fp32_parameters_bf16_cuda_autocast')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    if not check['passed']:raise RuntimeError('A2 downstream numerical gate failed; preserve receipt')


if __name__=='__main__':main()
