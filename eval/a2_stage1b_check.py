"""User-authorized Stage1b gate deviation; original Stage1 gate stays unchanged.

Use only in the final six-model screen under the native supervisor. The small
sample batch comparison is diagnostic up to the new hard limits. The full P300
batch decision is separate and cannot be replaced by this gate.
"""
import argparse
import json
import math
from pathlib import Path


def gate_decision(manual_error, batch_error, mismatches, compared_tokens):
    if (not all(math.isfinite(x) and x >= 0 for x in (manual_error,batch_error))
            or type(compared_tokens) is not int or compared_tokens <= 0
            or type(mismatches) is not int or not 0 <= mismatches <= compared_tokens):
        raise ValueError('Invalid numerical gate measurements')
    rate=mismatches/compared_tokens
    return dict(passed=manual_error<=2e-5 and batch_error<=.1 and rate<=.001,
                manual_max_nats_per_token=manual_error,batch_max_nats_per_token=batch_error,
                greedy_token_mismatches=mismatches,compared_tokens=compared_tokens,
                greedy_token_mismatch_rate=rate,
                old_batch_limit_exceeded=batch_error>.01,
                thresholds=dict(manual_nats_per_token=2e-5,batch_nats_per_token=.1,
                                greedy_token_mismatch_rate=.001))


def token_predictions(model, requests, batch_size, max_length):
    """Diagnostic argmax vectors using the same sort, right padding and positions.

    Also return LL to verify that this diagnostic reproduces the actual scorer.
    Never used as production scores.
    """
    import torch
    from efficient_llm.a2.model import autocast
    prepared=[]
    for i,r in enumerate(requests):
        ids=((r['context'] or [0])+r['continuation'])[-(max_length+1):][:-1]
        prepared.append((i,ids,r['continuation']))
    prepared.sort(key=lambda row:-len(row[1]));predictions=[None]*len(prepared);scores=[None]*len(prepared)
    with torch.inference_mode():
        for start in range(0,len(prepared),batch_size):
            rows=prepared[start:start+batch_size]
            ids=torch.zeros((len(rows),len(rows[0][1])),dtype=torch.long,device=model.device)
            for j,(_,tokens,_) in enumerate(rows):ids[j,:len(tokens)]=torch.tensor(tokens,device=model.device)
            with autocast(model.device):
                logits=model(input_ids=ids,use_cache=False,
                    position_ids=torch.arange(ids.shape[1],device=ids.device).expand_as(ids)).logits
            for j,(i,tokens,target) in enumerate(rows):
                selected=logits[j,len(tokens)-len(target):len(tokens)].float()
                predictions[i]=selected.argmax(-1).tolist()
                wanted=torch.tensor(target,device=ids.device)
                scores[i]=float(selected.log_softmax(-1).gather(1,wanted[:,None]).sum())
            del logits
    return predictions,scores


def check_model(model, requests, max_length):
    from eval.a2_check import check_model as original_check
    from eval.a2_likelihood import score_requests
    original=original_check(model,requests,max_length=max_length,batch_size=2)
    predictions={};diagnostic_errors={}
    for batch in (1,2):
        predictions[batch],ll=token_predictions(model,requests,batch,max_length)
        actual=score_requests(model,requests,batch_size=batch,max_length=max_length)
        errors=[abs(x-y[0])/len(r['continuation']) for x,y,r in zip(ll,actual['scores'],requests)]
        diagnostic_errors[str(batch)]=max(errors)
        if max(errors)>2e-5:print('DIAGNOSTIC_FLAG token diagnostic does not reproduce native scorer',flush=True)
    count=sum(len(p) for p in predictions[1])
    mismatch=sum(x!=y for a,b in zip(predictions[1],predictions[2]) for x,y in zip(a,b))
    result=gate_decision(original['manual_max_nats_per_token'],
                         max(original['batch_max_nats_per_token'],original['production_max_nats_per_token']),mismatch,count)
    result.update(original_gate=original,token_diagnostic_ll_errors=diagnostic_errors,
        batch1_token_predictions=predictions[1],batch2_token_predictions=predictions[2],
        production_batch_size=2,full_parent_batch_decision_still_required=True)
    return result


def main():
    from eval.a2_check import select_requests
    from eval.a2_checkpoint import load_verified,verify_identity
    from eval.a2_request_bridge import digest
    from scripts.a2_remaining import read_locked
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('identity','identity-sha256','prepared','prepared-sha256','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Output exists')
    identity=read_locked(dict(path=a.identity,sha256=a.identity_sha256))
    prepared=read_locked(dict(path=a.prepared,sha256=a.prepared_sha256))
    indices,requests=select_requests(prepared)
    model=load_verified(identity,a.device)
    result=check_model(model,requests,prepared['protocol']['max_length'])
    verify_identity(identity)
    result.update(checkpoint_identity_sha256=digest(identity),prepared_sha256=digest(prepared),
                  selected_request_indices=indices,precision='fp32_parameters_bf16_cuda_autocast')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    if not result['passed']:print('REPORTED_GATE_FAILURE continue under user policy; passed remains false',flush=True)


if __name__=='__main__':main()
