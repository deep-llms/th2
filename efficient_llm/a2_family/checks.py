"""Actual native-model FT1/FT2 checks and parent/pre-pruning arrays."""
import argparse
import gc
from pathlib import Path
import torch
from transformers import set_seed
from .common import read,write,Checks
from .model import prune,document_attention
from .evaluate import load_model,score_docs

@torch.no_grad()
def attention_checks(model):
    # FP32/no autocast, the no-removal transform must preserve exact logits.
    device=next(model.parameters()).device
    ids=torch.tensor([[100,200,300,400,500,600]],device=device)
    before=model(ids,use_cache=False).logits
    prune(model,[])
    after=model(ids,use_cache=False).logits
    identity=torch.equal(before,after)
    starts=torch.tensor([[1,0,0,1,0,0]],device=device,dtype=torch.bool)
    mask,pos=document_attention(starts)
    packed=model(ids,attention_mask=mask,position_ids=pos,use_cache=False).logits
    altered=ids.clone();altered[:,:3]=torch.tensor([700,800,900],device=device)
    other=model(altered,attention_mask=mask,position_ids=pos,use_cache=False).logits
    independent=torch.equal(packed[:,3:],other[:,3:])
    alone=model(ids[:,3:],use_cache=False).logits
    reset_error=float((packed[:,3:]-alone).abs().max())
    return dict(no_removal_bitwise=identity,document_independent=independent,position_reset_max_error=reset_error,
                masking_supported=independent and reset_error<=1e-5)


def run(cfg,arm):
    from accelerate import PartialState
    state=PartialState();rank,world=state.process_index,state.num_processes
    root=Path(cfg['results_root']);out=root/'parents'/arm
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    state.wait_for_everyone();checks=Checks(out/f'tests-rank{rank}.json')
    set_seed(0);entry=next(x for x in cfg['models']['checkpoints'] if x['label']==arm)
    model,loading=load_model(entry,state.device)
    checks.record('complete_model_load',not any(loading.values()),loading)
    actual=checks.attempt('FT1_FT2_execution',lambda:attention_checks(model))
    checks.record('FT1_no_removal_bitwise',actual is not None and actual['no_removal_bitwise'],actual)
    checks.record('FT2_document_mask_reset',actual is not None and actual['masking_supported'],actual)
    before_flags=[bool(x.self_attn.use_rope) for x in model.model.layers]
    parent={k:checks.attempt('parent:'+k,lambda k=k:score_docs(model,Path(cfg['cx_root'])/k)) for k in ('tuning','final')}
    if arm=='k64':
        ref=cfg['k64_reference_nll'];value=parent['final']
        checks.record('FT3_breadth_reference',value is not None and abs(value['loss']-ref)<=1e-6,dict(reference=ref,actual=None if value is None else value['loss']))
    keep=prune(model,list(range(24,33)))
    checks.record('FT1_original_RoPE_NoPE',len(keep)==27 and [bool(x.self_attn.use_rope) for x in model.model.layers]==[before_flags[i] for i in keep],dict(keep=keep))
    pre={k:checks.attempt('pre:'+k,lambda k=k:score_docs(model,Path(cfg['cx_root'])/k)) for k in ('tuning','final')}
    allflags=[None]*world;torch.distributed.all_gather_object(allflags,[x for x in checks.items if not x['passed']])
    masked=torch.tensor(int(actual is not None and actual['masking_supported']),device=state.device)
    torch.distributed.all_reduce(masked,op=torch.distributed.ReduceOp.MIN)
    if rank==0:
        write(out/'result.json',dict(arm=arm,parent=parent,pre=pre,mask_documents=bool(masked.item()),flags=[f for fs in allflags for f in fs],
                                  kept=keep,revision=entry['commit'],checks=actual))
        for label,scores in [('parent',parent),('pre-BS1',pre)]:
            for split,value in scores.items():
                if value is not None:write(out/'perseq'/split/(label+'.json'),value)
    del model;gc.collect();torch.cuda.empty_cache();state.wait_for_everyone()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--arm',required=True);a=p.parse_args();run(read(a.config),a.arm)
