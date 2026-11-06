"""Document-parallel version of the A2 S per-document scoring procedure."""
from pathlib import Path
import math
import numpy as np
import torch
from torch.nn import functional as F
from efficient_llm.a2.model import autocast
from .common import read,write


def rank_world():
    if torch.distributed.is_initialized():return torch.distributed.get_rank(),torch.distributed.get_world_size()
    return 0,1

@torch.no_grad()
def score_docs(model, folder, limit=None):
    model.eval();device=next(model.parameters()).device;rank,world=rank_world()
    folder=Path(folder);tokens=np.load(folder/'tokens.npy',mmap_mode='r');offsets=np.load(folder/'offsets.npy')
    count=len(offsets)-1 if limit is None else min(limit,len(offsets)-1)
    rows=[];failure=None
    try:
        for i in range(rank,count,world):
            start,end=map(int,offsets[i:i+2]);ids=torch.tensor(np.array(tokens[start:end],dtype=np.int64),device=device)[None]
            n=max(ids.shape[1]-1,0);total=torch.zeros((),dtype=torch.float64,device=device)
            if n:
                with autocast(device):
                    hidden=model.model(ids,use_cache=False,return_dict=True).last_hidden_state
                    for off in range(0,n,128):
                        hi=min(off+128,n);logits=model.lm_head(hidden[:,off:hi]).float()
                        losses=F.cross_entropy(logits.flatten(0,1),ids[:,off+1:hi+1].flatten(),reduction='none')
                        total+=losses.double().sum()
            rows.append((i,float(total),n))
    except Exception as error:failure=repr(error)
    payload=dict(rows=rows,error=failure)
    if world>1:
        all_payloads=[None]*world;torch.distributed.all_gather_object(all_payloads,payload)
    else:all_payloads=[payload]
    errors=[(i,p['error']) for i,p in enumerate(all_payloads) if p['error']]
    if errors:raise RuntimeError('Synchronized document scoring failure: '+repr(errors))
    combined=sorted(row for p in all_payloads for row in p['rows'])
    if [x[0] for x in combined]!=list(range(count)):raise ValueError('Document coverage differs')
    sums=[x[1] for x in combined];counts=[x[2] for x in combined]
    if sum(counts)==0:raise ValueError('No prediction targets')
    if not all(math.isfinite(s) for s in sums):raise ValueError('Nonfinite document NLL')
    return dict(loss=sum(sums)/sum(counts),per_sequence_nll=[s/n if n else 0. for s,n in zip(sums,counts)],
                per_sequence_tokens=counts,tokens=sum(counts),sequences=count,
                precision='FP32 parameters, BF16 autocast decoder/head, FP32 CE, FP64 sum',
                zero_prediction_documents=[i for i,n in enumerate(counts) if not n],
                estimator='token weighted',unit='documents')


def load_model(entry, device, *, teacher=False):
    from transformers import AutoModelForCausalLM
    model,loading=AutoModelForCausalLM.from_pretrained(entry['path'],local_files_only=True,
               torch_dtype=torch.bfloat16 if teacher else torch.float32,attn_implementation='sdpa',use_cache=False,output_loading_info=True)
    return model.to(device).eval(),loading
