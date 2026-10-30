"""Canonical P300 parent scores plus argmax evidence for the full batch decision.

Production likelihoods still come from the unchanged Stage1 score_requests.
A read-only forward hook records continuation argmax tokens from those exact
forwards, allowing LAMBADA prediction flips even when both answers are wrong.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path


def capture_scores(model, requests, *, batch_size, length):
    from eval.a2_likelihood import score_requests
    order=sorted(range(len(requests)),key=lambda i:-(min(len(requests[i]['context'] or [0])+len(requests[i]['continuation']),length+1)-1))
    predictions=[None]*len(requests);position=0
    def capture(module, args, kwargs, output):
        nonlocal position
        indices=order[position:position+batch_size];ids=kwargs['input_ids']
        if ids.shape[0]!=len(indices):raise ValueError('Diagnostic forward/request alignment changed')
        for j,i in enumerate(indices):
            r=requests[i];inputs=((r['context'] or [0])+r['continuation'])[-(length+1):][:-1]
            if ids[j,:len(inputs)].tolist()!=inputs:raise ValueError('Observed scoring inputs differ')
            predictions[i]=output.logits[j,len(inputs)-len(r['continuation']):len(inputs)].argmax(-1).tolist()
        position+=len(indices)
    handle=model.register_forward_hook(capture,with_kwargs=True)
    try:result=score_requests(model,requests,batch_size=batch_size,max_length=length)
    finally:handle.remove()
    if position!=len(requests) or any(p is None for p in predictions):raise ValueError('Incomplete argmax capture')
    for r,p,score in zip(requests,predictions,result['scores']):
        if (p==r['continuation'])!=score[1]:raise ValueError('Argmax evidence disagrees with canonical greedy score')
    return result,predictions


def score_with_predictions(identity, prepared, *, batch_size, device):
    from eval.a2_checkpoint import load_verified,verify_identity
    from eval.a2_likelihood import score_requests
    from eval.a2_request_bridge import digest
    if (batch_size not in (1,2) or identity['dataset']!='fwe'
            or identity['parent']!='P300' or identity['kind']!='parent'
            or len(prepared['requests'])!=78259
            or prepared['request_sha256']!=digest(prepared['requests'])
            or prepared['protocol_sha256']!=digest(prepared['protocol'])):
        raise ValueError('Require the full fixed FWE P300 batch comparison')
    model=load_verified(identity,device)
    result,predictions=capture_scores(model,prepared['requests'],batch_size=batch_size,
        length=prepared['protocol']['max_length'])
    verify_identity(identity)
    result.update(batch_size=batch_size,prepared_sha256=digest(prepared),request_sha256=prepared['request_sha256'],
        checkpoint_identity=identity,checkpoint_identity_sha256=digest(identity),
        packages={k:version(k) for k in ('torch','transformers','tokenizers')})
    return result,dict(batch_size=batch_size,prepared_sha256=digest(prepared),
        request_sha256=prepared['request_sha256'],checkpoint_identity_sha256=digest(identity),
        predictions=predictions,definition='teacher-forced continuation argmax tokens from exact canonical scoring forwards')


def main():
    from scripts.a2_remaining import read_locked
    from eval.a2_run import write
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('identity','identity-sha256','prepared','prepared-sha256','output','predictions-output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--batch-size',type=int,choices=(1,2),required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if Path(a.output).exists() or Path(a.predictions_output).exists():raise ValueError('Output exists')
    result,predictions=score_with_predictions(read_locked(dict(path=a.identity,sha256=a.identity_sha256)),
        read_locked(dict(path=a.prepared,sha256=a.prepared_sha256)),batch_size=a.batch_size,device=a.device)
    write(a.output,result);write(a.predictions_output,predictions)


if __name__=='__main__':main()
