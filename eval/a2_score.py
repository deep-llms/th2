"""Native offline A2 likelihood scoring of a pinned CPU request artifact.

GPU ownership remains with the outer supervisor. This command never acquires,
stops or starts other jobs; it loads one verified model and writes one new file.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='offline')

from eval.a2_checkpoint import load_verified, verify_identity
from eval.a2_likelihood import score_requests
from eval.a2_request_bridge import digest
from scripts.a2_remaining import read_locked


def score_artifact(identity, prepared, *, batch_size=8, device='cpu'):
    protocol=prepared['protocol']
    if (prepared.get('schema')!=1 or prepared.get('request_sha256')!=digest(prepared['requests'])
            or prepared.get('protocol_sha256')!=digest(protocol)
            or protocol.get('seed')!=42 or protocol.get('num_fewshot')!=0):
        raise ValueError('Changed request artifact or evaluation protocol')
    model=load_verified(identity,device)
    scores=score_requests(model,prepared['requests'],batch_size=batch_size,max_length=protocol['max_length'])
    verify_identity(identity)
    scores.update(prepared_sha256=digest(prepared),request_sha256=prepared['request_sha256'],
        checkpoint_identity=identity,checkpoint_identity_sha256=digest(identity),
        packages={k:version(k) for k in ('torch','transformers','tokenizers')})
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity',required=True);p.add_argument('--identity-sha256',required=True)
    p.add_argument('--prepared',required=True);p.add_argument('--prepared-sha256',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();out=Path(a.output)
    if out.exists():p.error('Output exists')
    if a.batch_size<1:p.error('Batch size must be positive')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:p.error('One independent scoring worker required')
    identity=read_locked(dict(path=a.identity,sha256=a.identity_sha256))
    prepared=read_locked(dict(path=a.prepared,sha256=a.prepared_sha256))
    result=score_artifact(identity,prepared,batch_size=a.batch_size,device=a.device)
    out.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(result,allow_nan=False)+'\n'
    with out.open('x') as stream:stream.write(encoded)


if __name__=='__main__':main()
