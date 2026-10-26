"""Gated selective-decode correctness, actual timing and synthetic lookup evaluation."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import time
import torch
import torch.distributed as dist
from transformers import AutoTokenizer,StaticCache
from efficient_llm.context_attention import ContextPolicy
from efficient_llm.context_study import policy
from efficient_llm.decode_profile import trial
from efficient_llm.diagnostic import load_model,load_probe,write
from efficient_llm.diagnostic_data import sha256
from efficient_llm.lookup_probe import build
from efficient_llm.routers import load_routers


def context(model,name,config,routers):
    if name=='dense':return nullcontext()
    if name in ('local','fixed_groups','joint_router','independent_router'):
        return policy(model,name,config,routers)
    kwargs={k:config[k] for k in ('keep','window','sinks')}
    if name=='dense_kernel':
        masks=torch.ones(model.config.num_hidden_layers,model.config.num_key_value_heads,dtype=torch.bool)
    elif name=='fixed_kernel':masks=torch.tensor(config['static_masks']['fixed_groups'],dtype=torch.bool)
    elif name in ('joint_kernel','independent_kernel'):
        return ContextPolicy(model,mode='router',routers=routers,bank=int(name=='independent_kernel'),decode_backend='triton',**kwargs)
    else:raise ValueError('Unknown deployment method')
    return ContextPolicy(model,mode='reference_static',static_masks=masks,decode_backend='triton',**kwargs)


def prepare(a):
    if a.output.exists():raise ValueError('Fresh input bundle required')
    trusted=json.loads(a.contract.read_text())
    for name,expected in trusted['files'].items():
        if sha256(a.previous_run/name)!=expected:raise ValueError('Previous study identity changed: '+name)
    config=json.loads((a.previous_run/'supervised/run/study/context-config.json').read_text())
    for name,expected in config['model_files'].items():
        if sha256(a.model/name)!=expected:raise ValueError('Model identity changed: '+name)
    a.output.mkdir();(a.output/'probe').mkdir()
    for source,dest in [('supervised/run/study/context-config.json','context-config.json'),
                        ('supervised/run/study/routers.pt','routers.pt'),('probe/manifest.json','probe/manifest.json'),('probe/tokens.pt','probe/tokens.pt')]:
        shutil.copyfile(a.previous_run/source,a.output/dest)
    load_probe(a.output/'probe')
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    lookup=build(tokenizer);torch.save(lookup,a.output/'lookup.pt')
    write(a.output/'manifest.json',dict(status='passed',previous_run=str(a.previous_run),contract_sha256=sha256(a.contract),
        files={str(p.relative_to(a.output)):sha256(p) for p in a.output.rglob('*') if p.is_file()},lookup_cases=len(lookup['rows'])))
    print('DEPLOY_INPUTS_PREPARED',len(lookup['rows']),flush=True)


def load_inputs(a,device,dtype):
    manifest=json.loads((a.inputs/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if sha256(a.inputs/name)!=expected:raise ValueError('Input changed: '+name)
    config=json.loads((a.inputs/'context-config.json').read_text())
    for name,expected in config['model_files'].items():
        if sha256(a.model/name)!=expected:raise ValueError('Changed model: '+name)
    if sha256(a.inputs/'routers.pt')!=config['routers_sha256']:raise ValueError('Router mismatch')
    return load_model(a.model,device,dtype),config,load_routers(a.inputs/'routers.pt',device)


@torch.inference_mode()
def cached_logits(model,ids,name,config,routers,prefill):
    with context(model,name,config,routers):
        cache=StaticCache(config=model.config,max_cache_len=ids.shape[1]+4)
        model(ids[:,:prefill],past_key_values=cache,use_cache=True,logits_to_keep=1)
        rows=[]
        for t in range(prefill,ids.shape[1]):
            rows.append(model(ids[:,t:t+1],past_key_values=cache,use_cache=True).logits.float())
    return torch.cat(rows,1)


@torch.inference_mode()
def gate(a):
    from efficient_llm.selective_decode import decode
    torch.manual_seed(87);torch.set_float32_matmul_precision('highest')
    rows=[]
    for dtype in (torch.float32,torch.bfloat16):
        for length,capacity,batch in [(1,7,1),(17,37,2),(132,260,2),(511,600,1),(2049,2084,8),(8192,8228,8)]:
            q=torch.randn(batch,16,1,128,device='cuda',dtype=dtype)
            k=torch.randn(batch,8,capacity,128,device='cuda',dtype=dtype);v=torch.randn_like(k)
            # Padded cache poison must never affect a current valid query.
            k[:,:,length:]=float('nan');v[:,:,length:]=float('nan')
            selected=(torch.arange(batch*8,device='cuda').reshape(batch,8)%3)!=0
            actual=decode(q,k,v,selected,length=length,window=128,sinks=4,scale=128**-.5)
            positions=torch.arange(length,device='cuda');local=(positions<4)|(positions>=length-128)
            allowed=selected.repeat_interleave(2,1)[:,:,None,None] | local[None,None,None,:]
            expected=torch.nn.functional.scaled_dot_product_attention(q.float(),k[:,:,:length].float(),v[:,:,:length].float(),
                attn_mask=allowed,enable_gqa=True,scale=128**-.5).transpose(1,2).to(dtype)
            torch.testing.assert_close(actual,expected,atol=3e-5 if dtype==torch.float32 else .002,rtol=3e-4 if dtype==torch.float32 else .01)
            rows.append(dict(dtype=str(dtype),length=length,batch=batch,max_abs=(actual.float()-expected.float()).abs().max().item()))
    # Local heads must not read distant NaNs, even in a logically live cache.
    q=torch.randn(2,16,1,128,device='cuda');k=torch.randn(2,8,512,128,device='cuda');v=torch.randn_like(k)
    k[:,:,4:384]=float('nan');v[:,:,4:384]=float('nan')
    out=decode(q,k,v,torch.zeros(2,8,device='cuda',dtype=torch.bool),length=512,window=128,sinks=4,scale=128**-.5)
    if not torch.isfinite(out).all():raise ValueError('Local branch read distant poisoned cache')
    model,config,routers=load_inputs(a,torch.device('cuda'),torch.float32)
    _,data=load_probe(a.inputs/'probe');ids=data['evaluation'][:1,:64].cuda();errors={}
    short={**config,'window':8,'sinks':2}
    for ref,new in [('dense','dense_kernel'),('fixed_groups','fixed_kernel'),('joint_router','joint_kernel'),('independent_router','independent_kernel')]:
        expected=cached_logits(model,ids,ref,short,routers,32)
        actual=cached_logits(model,ids,new,short,routers,32)
        torch.testing.assert_close(actual,expected,atol=2e-3,rtol=3e-4)
        errors[new]=(actual-expected).abs().max().item()
    write(a.output,dict(status='passed',kernel_checks=rows,poisoned_cache_passed=True,fp32_model_errors=errors,
        input_manifest_sha256=sha256(a.inputs/'manifest.json')))
    print('SELECTIVE_GATE_PASSED',flush=True)


@torch.inference_mode()
def distributed(a):
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)));torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl')
    try:
        model,config,routers=load_inputs(a,device,torch.bfloat16)
        torch.set_float32_matmul_precision('highest')
        rows=[]
        if a.command=='profile':
            cases=[(1,2048),(8,2048),(1,8192),(8,8192)]
            assigned=[cases[rank%4]] if world>=4 else cases[rank::world]
            names=['dense','dense_kernel','fixed_groups','fixed_kernel','joint_router','joint_kernel']
            for batch,length in assigned:
                g=torch.Generator().manual_seed(42+batch+length)
                ids=torch.randint(model.config.vocab_size,(batch,length+36),generator=g).to(device)
                measurements={n:[] for n in names}
                for repeat in range(-1,3):
                    shift=(rank+repeat+1)%len(names)
                    for name in names[shift:]+names[:shift]:
                        with context(model,name,config,routers):result=trial(model,ids,length,32)
                        if repeat>=0:measurements[name].append(result)
                medians={n:statistics.median(v['wall_ms'] for v in values)/32 for n,values in measurements.items()}
                rows += [dict(rank=rank,batch=batch,context=length,method=n,ms_per_step=medians[n],
                    speedup_vs_native=medians['dense']/medians[n],speedup_vs_dense_kernel=medians['dense_kernel']/medians[n],
                    repetitions=measurements[n]) for n in names]
            # BF16 own-trajectory check on real data, separate from synthetic timing.
            _,tokens=load_probe(a.inputs/'probe');ids=tokens['evaluation'][rank:rank+1,:].to(device)
            compare={};target=ids[:,2017:2049]
            for name in ('dense','dense_kernel','joint_router','joint_kernel'):
                scores=cached_logits(model,ids[:,:2048],name,config,routers,2016)
                nll=torch.nn.functional.cross_entropy(scores.flatten(0,1),target.flatten(),reduction='none')
                compare[name]=dict(nll=nll.cpu().tolist(),predictions=scores.argmax(-1).flatten().cpu().tolist())
            for ref,new in [('dense','dense_kernel'),('joint_router','joint_kernel')]:
                delta=abs(statistics.mean(compare[new]['nll'])-statistics.mean(compare[ref]['nll']))
                if delta>.01:raise ValueError(f'BF16 cached NLL discrepancy {new}: {delta}')
            extra=dict(rank=rank,bf16_cached_real_document=compare)
        else:
            lookup=torch.load(a.inputs/'lookup.pt',weights_only=True,map_location='cpu')
            candidates=torch.tensor(lookup['candidate_ids'],device=device)
            names=['dense','local','fixed_groups','joint_router','independent_router','joint_kernel']
            for i in range(rank,len(lookup['rows']),world):
                item=lookup['rows'][i];ids=item['tokens'][None].to(device)
                for name in names:
                    scores=cached_logits(model,ids,name,config,routers,ids.shape[1]-1)[0,0]
                    values=scores[candidates];label=item['label']
                    rows.append(dict(case=i,method=name,length=item['length'],depth=item['depth'],sample=item['sample'],
                        label=label,prediction=values.argmax().item(),correct=values.argmax().item()==label,
                        choice_logprobs=values.log_softmax(-1).cpu().tolist(),correct_vocab_nll=-scores.log_softmax(-1)[candidates[label]].item()))
            extra=None
        parts=[None]*world
        if world>1:dist.all_gather_object(parts,(rows,extra))
        else:parts=[(rows,extra)]
        if rank==0:
            results=[r for part,_ in parts for r in part]
            report=dict(status='passed',world_size=world,finished_utc=datetime.now(timezone.utc).isoformat(),
                input_manifest_sha256=sha256(a.inputs/'manifest.json'),results=results,extra=[x for _,x in parts if x])
            if a.command=='retrieval':
                if len(results)!=len(lookup['rows'])*len(names) or len({(r['case'],r['method']) for r in results})!=len(results):raise ValueError('Lookup coverage differs')
                report['summary']={}
                for name in names:
                    report['summary'][name]={}
                    for length in (2048,8192):
                        for depth in ('near','10%','50%','90%'):
                            cell=[r for r in results if (r['method'],r['length'],r['depth'])==(name,length,depth)]
                            report['summary'][name][f'{length}_{depth}']=dict(correct=sum(r['correct'] for r in cell),count=len(cell),accuracy=statistics.mean(r['correct'] for r in cell))
                report['notes']=lookup['notes']+['Synthetic eight-choice lookup; chance 12.5%; not free generation or a standard benchmark.']
            write(a.output,report);print('DEPLOY_RESULT',a.command,json.dumps(report.get('summary',results)),flush=True)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','gate','profile','retrieval'])
    for name in ('model','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('inputs','previous-run','contract'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Output must be fresh')
    if a.command=='prepare':prepare(a)
    elif a.command=='gate':gate(a)
    else:distributed(a)


if __name__=='__main__':main()
