"""Real-width forward/backward, FA4/SDPA and micro-16 capacity; owned GPUs only."""
import argparse
import gc
import math
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from datasets import Dataset
from bdot.model import BDOTForCausalLM
from packed_fa4 import register_backend
from packed_training import PackedCollator,model_inputs
from scripts.prepare_stop_adaptation import read,write
from eval.stop_study import fit_head,cached_policy_check


def inputs(ids, backend):
    rows=[dict(input_ids=r,labels=r,attention_mask=[1]*len(r),segments=[0]*len(r)) for r in ids]
    return model_inputs({k:v.cuda() if isinstance(v,torch.Tensor) else v
                        for k,v in PackedCollator(attention_backend=backend)(rows).items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);backend=register_backend();results={}
    for name in read(a.root/'branches.json'):
        init=a.root/'prepared'/(name+'-init')
        m=BDOTForCausalLM.from_pretrained(init,attn_implementation=backend).cuda().train()
        assert int(m.completed_steps)==0 and not any(p.requires_grad for p in m.stop_controller.parameters())
        ids=torch.randint(100,5000,(2,64),generator=torch.Generator().manual_seed(12)).tolist()
        losses=[];gradient_norms=[]
        for attention in ('fa4','sdpa'):
            m.set_attn_implementation(backend if attention=='fa4' else 'sdpa')
            m.routing_counter.zero_();m.load_counts.zero_();m.metric_totals.zero_();m.zero_grad(set_to_none=True)
            if hasattr(m,'exit_options'):m.exit_metric_totals.zero_()
            with torch.autocast('cuda',dtype=torch.bfloat16):out=m(**inputs(ids,attention))
            assert torch.isfinite(out.loss)
            out.loss.backward()
            loss=float(out.loss.detach());losses.append(loss)
            grad=torch.nn.utils.clip_grad_norm_(m.parameters(),float('inf'),error_if_nonfinite=True)
            gradient_norms.append(float(grad))
            assert int(m.routing_counter)==1
            del out
        assert abs(losses[0]-losses[1])<.04,(name,losses)
        assert abs(gradient_norms[0]-gradient_norms[1])/max(gradient_norms)<.1,(name,gradient_norms)
        m.set_attn_implementation(backend);m.zero_grad(set_to_none=True)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        # Same sequence/microbatch as the real run, fresh local optimizer. The
        # init files are immutable; smoke optimizer updates are never saved.
        large=inputs(torch.randint(100,5000,(16,1024),generator=torch.Generator().manual_seed(14)).tolist(),'fa4')
        optimizer=torch.optim.AdamW(m.parameters(),lr=3e-5,foreach=False)
        times=[]
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):out=m(**large)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
            del out
        peak=torch.cuda.max_memory_allocated();total=torch.cuda.get_device_properties(0).total_memory
        if peak>.85*total:raise RuntimeError(f'Insufficient micro16 memory headroom: {name} {peak}/{total}')
        del optimizer,large
        m.zero_grad(set_to_none=True);m.eval();m.set_attn_implementation('sdpa')
        m.stop_controller.policy='fixed';m.stop_controller.fixed_depth=6
        checks=cached_policy_check(m,dict(policy='fixed',fixed_depth=6),ids[0])
        assert checks['passed'],(name,checks)
        m.set_attn_implementation(backend)
        rows=[dict(input_ids=r,labels=r,attention_mask=[1]*len(r),segments=[0]*len(r)) for r in ids]
        head=fit_head(m,Dataset.from_list(rows),'fa4',size=2,samples_per_row=8,epochs=2)
        assert head['examples']==16 and all(math.isfinite(x) for x in head['bce_by_epoch'])
        assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('stop_controller.'))
        results[name]=dict(loss_fa4=losses[0],loss_sdpa=losses[1],gradient_norms=gradient_norms,
            micro16_forward_backward_optimizer_seconds=statistics.median(times[1:]),
            peak_allocated_bytes=peak,device_total_bytes=total,cached_check=checks,head_fit=head)
        del m;gc.collect();torch.cuda.empty_cache()
    write(a.output,dict(status='ok',branches=results,
        note='Synthetic numerical/capacity gate; wall time is not a real-data training ETA.'))


if __name__=='__main__':main()
