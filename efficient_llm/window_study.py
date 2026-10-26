"""Bounded sink/window ablation and compact decode scheduling screen. No training."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist
from transformers import StaticCache
from efficient_llm.context_attention import ContextPolicy
from efficient_llm.context_deploy import load_inputs
from efficient_llm.decode_profile import trial
from efficient_llm.diagnostic import load_probe,evaluate,paired_summary,write
from efficient_llm.diagnostic_data import sha256
from efficient_llm.selective_decode import decode,decode_compact


# Same fixed allocations for zero/four sinks: no selection on evaluation data.
METHODS={
    'dense_sdpa':('dense','sdpa',4),
    'dense_original':('dense','triton',4),
    'dense_compact':('dense','triton_compact',4),
    **{f'fixed_{backend}_s{sinks}':('fixed',impl,sinks)
       for backend,impl in [('sdpa','sdpa'),('original','triton'),('compact','triton_compact')]
       for sinks in (4,0)},
    'joint_original':('joint','triton',4),
    'joint_compact':('joint','triton_compact',4),
}


def context(model,name,config,routers):
    kind,backend,sinks=METHODS[name]
    if name=='dense_sdpa':return nullcontext()
    kwargs=dict(keep=config['keep'],window=config['window'],sinks=sinks,decode_backend=backend)
    if kind=='joint':return ContextPolicy(model,mode='router',routers=routers,**kwargs)
    masks=(torch.ones(model.config.num_hidden_layers,model.config.num_key_value_heads,dtype=torch.bool)
           if kind=='dense' else torch.tensor(config['static_masks']['fixed_groups'],dtype=torch.bool))
    return ContextPolicy(model,mode='static' if backend=='sdpa' else 'reference_static',static_masks=masks,**kwargs)


@torch.inference_mode()
def cached(model,ids,name,config,routers,prefill):
    with context(model,name,config,routers):
        cache=StaticCache(config=model.config,max_cache_len=ids.shape[1]+4)
        model(ids[:,:prefill],past_key_values=cache,use_cache=True,logits_to_keep=1)
        return torch.cat([model(ids[:,t:t+1],past_key_values=cache,use_cache=True).logits.float()
                          for t in range(prefill,ids.shape[1])],1)


@torch.inference_mode()
def gate(a):
    torch.manual_seed(87);torch.set_float32_matmul_precision('highest');rows=[]
    for dtype in (torch.float32,torch.bfloat16):
        for length,capacity,batch in [(1,7,1),(17,37,2),(259,600,2),(2049,2084,8),(8192,8228,8)]:
            q=torch.randn(batch,16,1,128,device='cuda',dtype=dtype)
            k=torch.randn(batch,8,capacity,128,device='cuda',dtype=dtype);v=torch.randn_like(k)
            k[:,:,length:]=float('nan');v[:,:,length:]=float('nan')
            for keep in (0,4,8):
                selected=(torch.arange(8,device='cuda')[None,:]+torch.arange(batch,device='cuda')[:,None])%8<keep
                for sinks in (0,4):
                    options=dict(length=length,window=128,sinks=sinks,scale=128**-.5)
                    actual=decode_compact(q,k,v,selected,global_groups=keep,**options)
                    original=decode(q,k,v,selected,**options)
                    positions=torch.arange(length,device='cuda')
                    allowed=selected.repeat_interleave(2,1)[:,:,None,None]|((positions<sinks)|(positions>=length-128))[None,None,None,:]
                    expected=torch.nn.functional.scaled_dot_product_attention(q.float(),k[:,:,:length].float(),v[:,:,:length].float(),
                        attn_mask=allowed,enable_gqa=True,scale=128**-.5).transpose(1,2).to(dtype)
                    for reference in (expected,original):
                        torch.testing.assert_close(actual,reference,atol=3e-5 if dtype==torch.float32 else .002,
                                                   rtol=3e-4 if dtype==torch.float32 else .01)
                    rows.append(dict(dtype=str(dtype),length=length,batch=batch,keep=keep,sinks=sinks,
                        reference_max_abs=(actual.float()-expected.float()).abs().max().item(),
                        original_max_abs=(actual.float()-original.float()).abs().max().item()))
    # No reads from inaccessible live history, and malformed masks fail visibly.
    q=torch.randn(2,16,1,128,device='cuda');k=torch.randn(2,8,512,128,device='cuda');v=torch.randn_like(k)
    k[:,:,4:384]=float('nan');v[:,:,4:384]=float('nan')
    selected=torch.zeros(2,8,device='cuda',dtype=torch.bool)
    for sinks in (0,4):
        actual=decode_compact(q,k,v,selected,length=512,window=128,sinks=sinks,scale=128**-.5,global_groups=0)
        if not torch.isfinite(actual).all():raise ValueError('Read inaccessible poisoned history')
    invalid=decode_compact(q,k,v,selected,length=512,window=128,sinks=4,scale=128**-.5,global_groups=4)
    if not torch.isnan(invalid).all():raise ValueError('Invalid mask budget must produce NaN')
    model,config,routers=load_inputs(a,torch.device('cuda'),torch.float32)
    _,data=load_probe(a.inputs/'probe');ids=data['evaluation'][:1,:192].cuda();errors={}
    for ref,new in [('dense_sdpa','dense_compact'),('fixed_sdpa_s4','fixed_compact_s4'),
                    ('fixed_sdpa_s0','fixed_compact_s0'),('joint_original','joint_compact')]:
        expected=cached(model,ids,ref,config,routers,160)
        actual=cached(model,ids,new,config,routers,160)
        torch.testing.assert_close(actual,expected,atol=.002,rtol=3e-4)
        errors[new]=(actual-expected).abs().max().item()
    write(a.output,dict(status='passed',kernel_checks=rows,fp32_model_errors=errors,
        input_manifest_sha256=sha256(a.inputs/'manifest.json'),
        note='Math/FP32 gate only; original BF16 native acceptance failure remains unresolved.'))
    print('WINDOW_COMPACT_GATE_PASSED',flush=True)


@torch.inference_mode()
def microbench(batch,length,device):
    torch.manual_seed(917+batch+length)
    q=torch.randn(batch,16,1,128,device=device,dtype=torch.bfloat16)
    k=torch.randn(batch,8,length+36,128,device=device,dtype=q.dtype);v=torch.randn_like(k)
    all_groups=torch.ones(batch,8,device=device,dtype=torch.bool)
    fixed=(torch.arange(8,device=device)[None,:]+torch.arange(batch,device=device)[:,None])%8<4
    results=[]
    for name in ('dense_original','dense_compact','fixed_original_s4','fixed_original_s0','fixed_compact_s4','fixed_compact_s0'):
        kind,backend,sinks=METHODS[name];selected=all_groups if kind=='dense' else fixed
        kwargs=dict(length=length,window=128,sinks=sinks,scale=128**-.5)
        func=decode_compact if backend=='triton_compact' else decode
        if func is decode_compact:kwargs['global_groups']=8 if kind=='dense' else 4
        def call():return func(q,k,v,selected,**kwargs)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):eager=call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        # Only this fixed-shape attention microbenchmark is captured. The
        # whole-model measurement remains ordinary eager cached decoding.
        with torch.cuda.graph(graph):
            for _ in range(8):output=call()
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(output,eager,atol=0,rtol=0)
        values=[]
        for _ in range(3):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter();start.record()
            for _ in range(100):graph.replay()
            end.record();torch.cuda.synchronize()
            values.append(dict(cuda_us_per_call=start.elapsed_time(end)*1000/800,
                               wall_us_per_call=(time.perf_counter()-wall)*1e6/800))
        old_tiles=batch*16*((length+36+255)//256)
        keep=8 if kind=='dense' else 4;splits=(length+255)//256
        results.append(dict(method=name,batch=batch,context=length,repetitions=values,
            median_cuda_us=statistics.median(r['cuda_us_per_call'] for r in values),
            partial_programs=old_tiles if func is decode else batch*(keep*2*splits+(8-keep)*2),
            merge_programs=batch*16 if func is decode else batch*keep*2))
        del graph,output,eager
    return results


@torch.inference_mode()
def profile(a,model,config,routers,device,rank,world):
    if world!=8:raise ValueError('This screen requires eight ranks/two replicates per shape')
    batch,length=[(1,2048),(8,2048),(1,8192),(8,8192)][rank%4]
    generator=torch.Generator().manual_seed(42+batch+length)
    ids=torch.randint(model.config.vocab_size,(batch,length+36),generator=generator).to(device)
    names=list(METHODS);values={name:[] for name in names}
    for repeat in range(-1,3):
        shift=(rank+repeat+1)%len(names)
        for name in names[shift:]+names[:shift]:
            with context(model,name,config,routers):result=trial(model,ids,length,32)
            if repeat>=0:values[name].append(result)
    timings=[dict(method=name,rank=rank,batch=batch,context=length,repetitions=v,
                  ms_per_step=statistics.median(r['wall_ms'] for r in v)/32) for name,v in values.items()]
    micro=microbench(batch,length,device)
    _,data=load_probe(a.inputs/'probe');ids=data['evaluation'][rank:rank+1].to(device)
    numerical={}
    for ref,new in [('dense_original','dense_compact'),('fixed_original_s4','fixed_compact_s4'),
                    ('fixed_original_s0','fixed_compact_s0'),('joint_original','joint_compact')]:
        expected=cached(model,ids[:,:2048],ref,config,routers,2016)
        actual=cached(model,ids[:,:2048],new,config,routers,2016)
        target=ids[:,2017:2049].flatten()
        nll=lambda x:torch.nn.functional.cross_entropy(x.flatten(0,1),target,reduction='none')
        left,right=nll(expected),nll(actual)
        numerical[new]=dict(reference=ref,max_abs_logit=(actual-expected).abs().max().item(),
            nll_delta=(right-left).mean().item(),original_nll=left.cpu().tolist(),compact_nll=right.cpu().tolist(),
            changed_predictions=(actual.argmax(-1)!=expected.argmax(-1)).sum().item())
    result=dict(rank=rank,timings=timings,attention_microbenchmark=micro,bf16_comparisons=numerical)
    write(a.output.with_name(a.output.stem+f'-rank{rank}.json'),result)
    return result


@torch.inference_mode()
def quality(a,model,config,routers,rank,world):
    manifest,data=load_probe(a.inputs/'probe');results={}
    for name in ('dense_sdpa','fixed_sdpa_s4','fixed_sdpa_s0'):
        selection=None if name=='dense_sdpa' else context(model,name,config,routers)
        metrics,_=evaluate(model,data['evaluation'],keep=config['keep'],policy='dense',selection=selection,
                           batch_size=1,rank=rank,world=world)
        results[name]=metrics
        if rank==0:print('WINDOW_QUALITY',name,json.dumps({k:v for k,v in metrics.items() if k!='per_document_nll'}),flush=True)
    results['zero_vs_four_sinks']=paired_summary(results['fixed_sdpa_s0']['per_document_nll'],results['fixed_sdpa_s4']['per_document_nll'])
    return dict(methods=results,probe_manifest_sha256=sha256(a.inputs/'probe/manifest.json'),
        note='Native SDPA only; original fixed allocations retained for both masks. No retraining, no kernel quality claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['gate','profile','quality'])
    for name in ('model','inputs','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Require fresh output')
    if a.command=='gate':gate(a);return
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)));torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl')
    try:
        model,config,routers=load_inputs(a,device,torch.bfloat16)
        torch.set_float32_matmul_precision('highest')
        if a.command=='profile':
            result=profile(a,model,config,routers,device,rank,world);parts=[None]*world
            if world>1:dist.all_gather_object(parts,result)
            else:parts=[result]
            result=dict(results=parts,diagnostic_only=True,native_backend_accepted=False,
                notes=['Whole-model eager timings include all overhead; prefill/JIT excluded.',
                       'Attention-only graph captures eight calls per replay; not a whole-model graph speed claim.',
                       'Existing BF16 native-trajectory gate remains failed; this tests scheduling and mask costs.'])
        else:result=quality(a,model,config,routers,rank,world)
        if rank==0:
            write(a.output,dict(status='measured' if a.command=='profile' else 'passed',world_size=world,
                finished_utc=datetime.now(timezone.utc).isoformat(),input_manifest_sha256=sha256(a.inputs/'manifest.json'),**result))
            print('WINDOW_STUDY_COMPLETE',a.command,flush=True)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
