"""Offline numerical gate for sink/recent context policies, on actual model weights."""
import argparse
from pathlib import Path
import torch
from transformers import StaticCache, DynamicCache
from efficient_llm.context_attention import ContextPolicy
from efficient_llm.diagnostic import load_model, load_probe, write
from efficient_llm.routers import RouterPair


@torch.inference_mode()
def check(model, ids, routers=None):
    groups=model.config.num_key_value_heads; keep=groups//2
    layers=model.config.num_hidden_layers; device=ids.device
    masks=torch.zeros(layers,groups,dtype=torch.bool,device=device)
    for i in range(layers): masks[i,torch.arange(keep,device=device).add(i).remainder(groups)]=True
    if routers is None:
        routers=RouterPair(layers,model.config.hidden_size,groups,keep,torch.ones(layers)).to(device).eval()
    kwargs=dict(keep=keep,window=8,sinks=2)
    errors={}
    def close(name, actual, expected):
        torch.testing.assert_close(actual,expected,rtol=3e-4,atol=2e-3)
        errors[name]=(actual-expected).abs().max().item()
    dense=model(ids,use_cache=False).logits
    with ContextPolicy(model,mode='teacher',stride=4,**kwargs) as teacher:
        close('teacher_identity',model(ids,use_cache=False).logits,dense)
        x,cost=teacher.take()
        if not torch.isfinite(x).all() or not torch.isfinite(cost).all():raise ValueError('Nonfinite teacher')
    with ContextPolicy(model,mode='reference_static',static_masks=masks,**kwargs):
        reference=model(ids,use_cache=False).logits
    with ContextPolicy(model,mode='static',static_masks=masks,**kwargs):
        close('static_execution',model(ids,use_cache=False).logits,reference)
    for mode,extra in [('local',{}),('static',dict(static_masks=masks)),
                       ('oracle',{}),('router',dict(routers=routers,bank=0)),
                       ('router',dict(routers=routers,bank=1))]:
        name=mode+str(extra.get('bank',''))
        with ContextPolicy(model,mode=mode,**extra,**kwargs):
            full=model(ids,use_cache=False).logits
            prefix=model(ids[:,:ids.shape[1]//2],use_cache=False).logits
            close(name+'_causal',prefix,full[:,:prefix.shape[1]])
            for kind in ('dynamic','static'):
                cache=DynamicCache(config=model.config) if kind=='dynamic' else StaticCache(config=model.config,max_cache_len=ids.shape[1]+3)
                cut=ids.shape[1]//2
                pieces=[model(ids[:,:cut],past_key_values=cache,use_cache=True).logits]
                for t in range(cut,ids.shape[1]):
                    pieces.append(model(ids[:,t:t+1],past_key_values=cache,use_cache=True).logits)
                close(name+'_'+kind,torch.cat(pieces,1),full)
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','probe','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Refuse existing check output')
    torch.manual_seed(42);torch.set_float32_matmul_precision('highest')
    manifest,tokens=load_probe(a.probe)
    from efficient_llm.diagnostic_data import sha256
    for name,expected in manifest['model_files'].items():
        if sha256(a.model/name)!=expected:raise ValueError('Model identity changed')
    model=load_model(a.model,torch.device('cuda'),torch.float32)
    ids=tokens['calibration'][:1,:32].cuda()
    errors=check(model,ids)
    model.bfloat16()
    # BF16 checks explicitly separate numerical precision from exact FP32 cache semantics.
    with torch.inference_mode(),ContextPolicy(model,mode='local',window=8,sinks=2):
        logits=model(ids,use_cache=False).logits
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite BF16 local logits')
    write(a.output,dict(status='passed',fp32_errors=errors,bf16_finite=True,backend='native_sdpa'))
    print('CONTEXT_GPU_CHECK_PASSED',flush=True)


if __name__=='__main__':main()
