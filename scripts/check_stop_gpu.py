"""Gate STOP on the real checkpoint: dense oracle, cache and measured headroom.

Runs only under an externally owned GPU allocation; never manages GPU workers.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer
from bdot.model import BDOTForCausalLM
from bdot.stopping import StopController
from packed_training import PackedCollator, model_inputs
from packed_fa4 import register_backend


def setup_stop(m, probability=0.):
    from dataclasses import replace
    m.options = replace(m.options, curriculum='none')
    m.config.bdot = m.options.to_dict()
    for layer in m.routed_steps: layer.options = m.options
    m.config.bdot_stop = dict(min_steps=2, train_random_prob=probability)
    m.stop_controller = StopController(m.config, m.options).to(m.device)


def held_reference(m, depths):
    """Dense layer oracle, including actual FiLM and weights, without copying."""
    from contextlib import contextmanager
    @contextmanager
    def reference():
        controller = m.stop_controller
        del m.stop_controller
        originals = []
        try:
            for layer in m.routed_steps:
                originals.append(layer.forward)
                orig, step = layer._forward_full, layer.step
                def held(h, orig=orig, step=step, **kw):
                    return torch.where((depths > step)[..., None], orig(h, **kw), h)
                layer.forward = held
            yield
        finally:
            for layer, orig in zip(m.routed_steps, originals): layer.forward = orig
            m.stop_controller = controller
    return reference()


def timing(m, batch, depth, context=128, repeats=32):
    m.stop_controller.policy = 'none' if depth == 8 else 'fixed'
    m.stop_controller.fixed_depth = depth
    # Identical supplied token stream across policies; no sampling-time noise.
    ids = torch.full((batch, context + repeats + 4), 250, device='cuda', dtype=torch.long)
    records = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(3):
            torch.cuda.synchronize(); start = time.perf_counter()
            result = m(input_ids=ids[:,:context], use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(); prefill = time.perf_counter() - start
            cache = result.past_key_values
            # Warm kernels and allocator, then time full model incl LM head.
            for t in range(context, context+4):
                cache = m(input_ids=ids[:,t:t+1], past_key_values=cache, use_cache=True).past_key_values
            torch.cuda.synchronize(); start = time.perf_counter()
            for t in range(context+4, ids.shape[1]):
                cache = m(input_ids=ids[:,t:t+1], past_key_values=cache, use_cache=True).past_key_values
            torch.cuda.synchronize()
            records.append(dict(prefill_seconds=prefill, decode_ms=1000*(time.perf_counter()-start)/repeats))
    import statistics
    return dict(batch=batch, fixed_depth=depth, context=context,
        decode_ms=statistics.median(r['decode_ms'] for r in records), trials=records)


def check(checkpoint):
    backend = register_backend()
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    ids = tok.encode('The city was quiet that morning. A bird flew over the river. '*5, add_special_tokens=False)[:64]
    ids = (ids * 2)[:64]
    rows = [dict(input_ids=ids, labels=ids, attention_mask=[1]*64,
                 segments=[0]*32+[1]*32) for _ in range(2)]
    def inputs(backend):
        x = PackedCollator(attention_backend=backend)(rows); x.pop('labels')
        return model_inputs({k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in x.items()})
    fa, sdpa = inputs('fa4'), inputs('sdpa')
    m = BDOTForCausalLM.from_pretrained(checkpoint, attn_implementation=backend).cuda().eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        original = m(**fa).logits
        setup_stop(m)
        none = m(**fa).logits
        torch.testing.assert_close(original, none, atol=0, rtol=0)
        depths = (torch.arange(128,device='cuda').reshape(2,64)%7+2).long()
        sparse = m(**fa, bdot_stop_depths=depths).logits
        with held_reference(m,depths): dense = m(**fa).logits
        # Gathered projection shapes may round differently in BF16.
        difference = float((sparse-dense).float().square().mean().sqrt())
        assert difference < .025, difference
        m.set_attn_implementation('sdpa')
        parallel = m(**sdpa, bdot_stop_depths=depths).logits
        cross_backend = float((sparse-parallel).float().square().mean().sqrt())
        assert cross_backend < .04, cross_backend
        cache = None; logits = []
        for t in range(64):
            result = m(input_ids=sdpa['input_ids'][:,t:t+1], position_ids=sdpa['position_ids'][:,t:t+1],
                attention_mask=sdpa['attention_mask'][:,:,t:t+1,:t+1], past_key_values=cache,
                use_cache=True, bdot_stop_depths=depths[:,t:t+1])
            cache = result.past_key_values; logits.append(result.logits)
            assert all(layer.get_seq_length()==t+1 for layer in cache.layers)
        recurrent = torch.cat(logits,1)
        cache_error = float((parallel-recurrent).float().square().mean().sqrt())
        assert cache_error < .04, cache_error
    records = [timing(m,b,d) for b in (1,32) for d in (8,2)]
    result = dict(status='ok',no_stop_max_abs=0., held_reference_logit_rms=difference,
        fa4_sdpa_logit_rms=cross_backend,cached_parallel_logit_rms=cache_error,timing=records,
        timing_backend='sdpa for all cached paths; FA4 for packed training',
        caveat='Fixed depth 2 is an optimistic speed bound, not a trained quality result.')
    del m
    torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    results={name:check(path) for name,path in json.loads(a.checkpoints.read_text()).items()}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(dict(status='ok',models=results),f,indent=2);f.write('\n')


if __name__=='__main__':main()
