"""GPU smoke for causal routing, frozen-teacher gradients and saved router reload."""
import argparse
from pathlib import Path
import torch
from efficient_llm.diagnostic import load_model,load_probe,write
from efficient_llm.routers import RouterPair,RouterSelection,TeacherFeatures,save_routers,load_routers


@torch.no_grad()
def cached_check(model,routers,ids):
    report={}
    for bank in (0,1):
        with RouterSelection(model,routers,bank):
            full=model(ids,use_cache=False).logits
            changed=ids.clone();changed[:,ids.shape[1]//2:]=1
            future=model(changed,use_cache=False).logits
            torch.testing.assert_close(full[:,:ids.shape[1]//2],future[:,:ids.shape[1]//2],rtol=0,atol=0)
            cache,parts=None,[]
            for t in range(ids.shape[1]):
                out=model(ids[:,t:t+1],past_key_values=cache,use_cache=True)
                cache=out.past_key_values;parts.append(out.logits)
            cached=torch.cat(parts,1)
            torch.testing.assert_close(full,cached,rtol=3e-4,atol=2e-3)
            if any(layer.keys.shape[1]!=model.config.num_key_value_heads for layer in cache.layers):
                raise ValueError('KV groups missing')
            report[str(bank)]=dict(cached_max_abs=(full-cached).abs().max().item(),future_max_abs=0.)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Existing GPU check result')
    torch.manual_seed(42);torch.set_float32_matmul_precision('highest')
    model=load_model(a.model,'cuda:0',torch.float32)
    _,tokens=load_probe(a.probe);ids=tokens['evaluation'][:1,:24].cuda()
    with torch.no_grad(),TeacherFeatures(model,6,stride=4) as teacher:
        before=model(ids).logits
        x,costs=teacher.take()
    scales=costs[:,:,0].std(-1,unbiased=False).mean(0).clamp_min(1e-8)
    routers=RouterPair(model.config.num_hidden_layers,model.config.hidden_size,8,6,scales.cpu()).cuda()
    loss=routers(x,costs)['loss'];loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in routers.parameters()):
        raise ValueError('Router gradient failure')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Backbone received gradient')
    with torch.no_grad():torch.testing.assert_close(model(ids).logits,before,rtol=0,atol=0)
    results=cached_check(model,routers,ids)
    path=a.output.with_suffix('.pt');save_routers(routers,path);loaded=load_routers(path,'cuda:0')
    with torch.no_grad():torch.testing.assert_close(loaded.scores(x),routers.scores(x),rtol=0,atol=0)
    model.bfloat16()
    with torch.no_grad(),RouterSelection(model,loaded,0):
        logits=model(ids).logits
        changed=ids.clone();changed[:,12:]=1
        torch.testing.assert_close(logits[:,:12],model(changed).logits[:,:12],rtol=0,atol=0)
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite BF16 routing')
    write(a.output,dict(status='passed',checks=results,frozen_backbone=True,finite_gradients=True,reload_exact=True,bf16_causal=True))
    print('ROUTER_CHECK_PASSED',flush=True)


if __name__=='__main__':main()
