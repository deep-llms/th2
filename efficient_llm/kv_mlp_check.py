"""Reference algebra and actual-weight CUDA gates for the KV/MLP pilot."""
import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from transformers import DynamicCache, AutoModelForCausalLM
from efficient_llm.kv_mlp import load_compressed, convert
from efficient_llm.diagnostic import load_probe, write
from efficient_llm.diagnostic_data import sha256


def explicit_attention(q, k, v, scale):
    """FP32 accumulation oracle; round SDPA inputs under autocast, not scores."""
    if torch.is_autocast_enabled(q.device.type):
        dtype=torch.get_autocast_dtype(q.device.type)
        q,k,v=(x.to(dtype) for x in (q,k,v))
    k=k.repeat_interleave(q.shape[1]//k.shape[1],1)
    v=v.repeat_interleave(q.shape[1]//v.shape[1],1)
    with torch.autocast(q.device.type,enabled=False):
        scores=(q.float()@k.float().transpose(-1,-2))*scale
        mask=torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril()
        probability=scores.masked_fill(~mask,-float('inf')).softmax(-1)
        return (probability@v.float()).to(q.dtype)


def fixed_input_checks(model, ids):
    """Do not amplify different kernels' rounding through 28 altered layers."""
    import efficient_llm.kv_mlp as implementation
    original=implementation.sdpa_attention_forward
    captured=[]
    def capture(module,q,k,v,mask,**kwargs):
        if mask is not None:raise ValueError('Gate requires a single causal prefix')
        captured.append((module.layer_idx,tuple(x.detach().clone() for x in (q,k,v)),module.scaling))
        return original(module,q,k,v,mask,**kwargs)
    with torch.no_grad(),torch.autocast(ids.device.type,dtype=torch.bfloat16):
        with patch.object(implementation,'sdpa_attention_forward',capture):
            model(ids,use_cache=False)
    rows=[]
    for layer,values,scale in captured:
        actual_inputs=[x.clone().requires_grad_(True) for x in values]
        expected_inputs=[x.clone().requires_grad_(True) for x in values]
        with torch.autocast(ids.device.type,dtype=torch.bfloat16):
            actual=F.scaled_dot_product_attention(*actual_inputs,is_causal=True,scale=scale,enable_gqa=True)
            expected=explicit_attention(*expected_inputs,scale)
            actual.float().square().mean().backward()
            expected.float().square().mean().backward()
        ga=torch.cat([x.grad.float().flatten() for x in actual_inputs])
        gb=torch.cat([x.grad.float().flatten() for x in expected_inputs])
        rows.append(dict(layer=layer,output_relative_l2=relative(actual,expected),
                         gradient_relative_l2=relative(ga,gb)))
    if len(rows)!=model.config.num_hidden_layers:raise ValueError('Incomplete attention coverage')
    return rows


def reference(model, ids, *, sdpa=False):
    """Independent decoder algebra: explicit softmax or the same SDPA primitive."""
    h = model.model.embed_tokens(ids)
    positions = torch.arange(ids.shape[1], device=ids.device)[None]
    cos, sin = model.model.rotary_emb(h, positions)
    shared = {}
    for i, layer in enumerate(model.model.layers):
        x = layer.input_layernorm(h); a = layer.self_attn
        shape = (*x.shape[:2], -1, a.head_dim)
        def rope(z):
            lo, hi = z.chunk(2, -1)
            return z*cos[:,None] + torch.cat((-hi,lo),-1)*sin[:,None]
        q = rope(a.q_norm(a.q_proj(x).view(shape)).transpose(1,2))
        if a.source_idx == i:
            k = rope(a.k_norm(a.k_proj(x).view(shape)).transpose(1,2))
            v = a.v_proj(x).view(shape).transpose(1,2)
            shared[i] = (k,v)
        k,v = shared[a.source_idx]
        if sdpa:
            # Independent decoder algebra, same kernel as production: separates
            # implementation errors from mixed-precision accumulation differences.
            y = F.scaled_dot_product_attention(q,k,v,is_causal=True,
                                               scale=a.scaling,enable_gqa=True)
        else:
            y=explicit_attention(q,k,v,a.scaling)
        h = h+a.o_proj(y.transpose(1,2).flatten(2))
        h = h+layer.mlp(layer.post_attention_layernorm(h))
    return model.lm_head(model.model.norm(h))


def relative(a, b):
    return float((a.detach().float()-b.detach().float()).norm()/b.detach().float().norm().clamp_min(1e-12))


def cache_bytes(cache):
    return sum(x.numel()*x.element_size() for layer in cache.layers if layer.is_initialized
               for x in (layer.keys,layer.values))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','probe','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sharing-layout', choices=['full50','uniform25','upper25'])
    a=p.parse_args()
    if a.output.exists():raise ValueError('Require fresh GPU gate output')
    manifest,tokens=load_probe(a.probe)
    for name,expected in manifest['model_files'].items():
        if sha256(a.model/name)!=expected:raise ValueError('Model identity mismatch')
    torch.set_float32_matmul_precision('highest'); torch.manual_seed(42)
    device=torch.device('cuda'); ids=tokens['calibration'][:1,:24].to(device)
    model=load_compressed(a.model,device).eval()
    native=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.float32,
                                               attn_implementation='sdpa').to(device).eval()
    errors={}
    with torch.no_grad():
        expected=native(ids,use_cache=False).logits
        actual=model(ids,use_cache=False).logits
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            torch.testing.assert_close(model(ids,use_cache=False).logits,
                                       native(ids,use_cache=False).logits,rtol=0,atol=0)
    del native,expected,actual
    channels=[list(range(model.config.intermediate_size*3//4))]*model.config.num_hidden_layers
    if a.sharing_layout:
        from efficient_llm.kv_placement import apply_layout
        apply_layout(model,a.sharing_layout)
    else:
        convert(model,share=True,channels=channels)
    ref=deepcopy(model)
    failed=[]
    for precision in ('fp32','bf16','bf16_same_kernel'):
        model.zero_grad(set_to_none=True);ref.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision!='fp32'):
            actual=model(ids,use_cache=False).logits
            expected=reference(ref,ids,sdpa=precision=='bf16_same_kernel')
            actual.float().square().mean().backward()
            expected.float().square().mean().backward()
        errors[precision+'_logits_relative_l2']=relative(actual,expected)
        numerator=denominator=torch.zeros((),device=device,dtype=torch.float64)
        # Rebind rather than in-place: these two accumulators initially alias.
        for (name,x),(other,y) in zip(model.named_parameters(),ref.named_parameters()):
            if name!=other or x.grad is None or y.grad is None:raise ValueError('Missing parameter gradient: '+name)
            numerator=numerator+(x.grad.float()-y.grad.float()).square().double().sum()
            denominator=denominator+y.grad.float().square().double().sum()
        errors[precision+'_gradient_relative_l2']=float((numerator/denominator).sqrt())
        limit=.03 if precision=='bf16' else 3e-4
        # Full-model cross-kernel BF16 drift remains a diagnostic, NOT a claim
        # of interchangeability. Gate BF16 explicit math at identical inputs
        # below, and gate the independent full decoder using the same kernel.
        if precision!='bf16' and (errors[precision+'_logits_relative_l2']>limit or errors[precision+'_gradient_relative_l2']>2*limit):
            failed.append(precision)
        print('KV_MLP_NUMERICAL',precision,errors,flush=True)
    del ref,expected,actual
    model.zero_grad(set_to_none=True)
    attention=fixed_input_checks(model,ids)
    if any(row['output_relative_l2']>.03 or row['gradient_relative_l2']>.06 for row in attention):
        failed.append('fixed_input_bf16_attention')
    if failed:
        write(a.output,dict(status='failed',failed=failed,errors=errors,attention=attention))
        raise ValueError('Reference mismatch: '+str(errors))
    with torch.no_grad():
        full=model(ids,use_cache=False).logits
        prefix=model(ids[:,:12],use_cache=False).logits
        torch.testing.assert_close(prefix,full[:,:12],rtol=3e-4,atol=2e-3)
        cache=DynamicCache(config=model.config)
        parts=[model(ids[:,:12],past_key_values=cache,use_cache=True).logits]
        for t in range(12,24):parts.append(model(ids[:,t:t+1],past_key_values=cache,use_cache=True).logits)
        torch.testing.assert_close(torch.cat(parts,1),full,rtol=3e-4,atol=2e-3)
        errors['cached_fp32_max_abs']=float((torch.cat(parts,1)-full).abs().max())
        active=sum(layer.is_initialized for layer in cache.layers)
        if active != len(set(model.config.kv_sources)):raise ValueError('Unexpected cache allocation')
        expected_bytes=active*2*ids.shape[0]*ids.shape[1]*model.config.num_key_value_heads*model.config.head_dim*4
        if cache_bytes(cache)!=expected_bytes:raise ValueError('Shared KV storage duplicated')
    write(a.output,dict(status='passed',errors=errors,attention=attention,
        cross_kernel_bf16_full_model_accepted=False,backend='native_sdpa',active_kv_layers=active,
        cache_bytes=cache_bytes(cache),parameters=sum(x.numel() for x in model.parameters()),
        model_files=manifest['model_files'],sharing_layout=a.sharing_layout))
    print('KV_MLP_GPU_GATE_PASSED',flush=True)


if __name__=='__main__':main()
