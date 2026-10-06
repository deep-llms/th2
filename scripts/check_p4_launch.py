"""Full-Qwen CUDA acceptance before the four-head P4 screen; no optimizer updates.

Compare the routed estimator with its detached-input recomputation reference,
including separate loss routes. Uses disposable packed diagnostic rows only;
production training continues to use train.py's normal dataset/cache path.
"""
import argparse
from contextlib import nullcontext
import copy
import gc
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig

from deep_kv.proxy import ProxyModel
from deep_kv.proxy_training import ProxyTrainer
from deep_kv.packing import isolated_data_collator
from scripts.check_fa4_proxy import read, write


def check(args):
    # Strict same-weight gradient comparisons must not include cuDNN's allowed
    # nondeterministic reduction noise. These settings affect this probe only.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    recipe=read(args.recipe)
    cfg=AutoConfig.from_pretrained(recipe['config_name'],local_files_only=True)
    cfg._attn_implementation='sdpa';cfg.use_cache=False
    rows=read(args.rows)[:2]
    assert len(rows)==2 and all(len(r['input_ids'])==2048 for r in rows)
    assert len(set(rows[0]['segments']))>1
    ctx=ProxyTrainer.context({k:v.cuda() for k,v in isolated_data_collator(rows).items()})
    results=[]
    for arm in args.arms:
        model=ProxyModel.from_scratch(cfg,arm,seed=42,checkpoint_layers=False,
            checkpoint_lm=False,checkpoint_aux=False,lm_chunk=128).cuda().train()
        assert model.value_groups==2 and model.settings.alpha_init==1
        with torch.no_grad():
            for mode in ('mean','variance'):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    out=model(ctx,compute_auxiliary_losses=False,auxiliary_grad=False,
                        collect_target_statistics=True,statistics_mode=mode)
                model.update_statistics(out['center_sums'],out['center_squares'],out['center_counts'],initialize=mode)
        buffers={k:v.clone() for k,v in model.named_buffers()}
        comparisons=[]
        for component in args.components:
            reference=None
            for capture_index,recompute in enumerate((True,True,False)):
                print('CAPTURE',arm,component,'recompute',recompute,flush=True)
                model.settings.aux_recompute=recompute;model.zero_grad(set_to_none=True)
                with torch.autograd.detect_anomaly() if args.anomaly else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        out=model(ctx)
                        lm=out['lm_sum']/out['lm_count'];aux=out['aux_sum']/out['aux_count']
                        loss={'lm':lm,'aux':aux,'combined':lm+.1*aux}[component]
                    loss.backward()
                gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
                invalid={n:dict(nan=int(g.isnan().sum()),inf=int(g.isinf().sum()))
                         for n,g in gradients.items() if not torch.isfinite(g).all()}
                if not torch.isfinite(loss) or invalid:
                    failure=dict(status='failed',arm=arm,component=component,recompute=recompute,
                        finite_loss=bool(torch.isfinite(loss)),lm_loss=str(float(lm.detach())),
                        aux_loss=str(float(aux.detach())),invalid_gradients=invalid)
                    write(args.output,failure)
                    print('NONFINITE_CAPTURE',json.dumps(failure),flush=True)
                    raise AssertionError('Nonfinite capture; tensor identities saved')
                for name,p in model.named_parameters():
                    estimator=name.startswith('heads.') and not name.endswith('.alpha')
                    active=p.grad is not None and bool(p.grad.count_nonzero())
                    if component=='aux':assert active==estimator,name
                    if component=='lm' and estimator:assert active==('iso' not in arm),name
                    if component=='lm' and name.endswith('.alpha'):assert active,name
                for name,value in model.named_buffers():assert torch.equal(value,buffers[name]),name
                if reference is None:
                    reference=gradients;reference_loss=float(loss.detach())
                else:
                    assert reference.keys()==gradients.keys()
                    errors={n:float((g-reference[n]).abs().max()/reference[n].abs().max().clamp_min(1e-20))
                            for n,g in gradients.items()}
                    worst=max(errors,key=errors.get)
                    entry=dict(component=component,reference_repeat=capture_index==1,
                        loss_difference=abs(float(loss.detach())-reference_loss),
                        maximum_relative_gradient_error=errors[worst],worst_parameter=worst)
                    comparisons.append(entry)
                    print(arm,json.dumps(entry),flush=True)
                    assert entry['loss_difference']<1e-5 and errors[worst]<=.01,entry
            del gradients,reference,out,loss,lm,aux
        model.settings.aux_recompute=False;model.zero_grad(set_to_none=True)
        captured=[]
        hook=model.backbone.model.embed_tokens.register_forward_hook(lambda mod,inp,out:captured.append(out))
        with torch.autocast('cuda',dtype=torch.bfloat16):hidden=model.hidden_states(ctx)[0]
        boundary=int((ctx.segments[0,1:]!=ctx.segments[0,:-1]).nonzero()[0])+1
        gradient=torch.autograd.grad(hidden[0,boundary:].float().square().sum(),captured[-1])[0]
        assert float(gradient[0,:boundary].abs().max())==0
        hook.remove();captured.clear()
        changed=copy.deepcopy(ctx);changed.input_ids=ctx.input_ids.clone();changed.input_ids[0,:boundary]=17
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after=model.hidden_states(changed)[0]
        assert torch.equal(hidden[0,boundary:],after[0,boundary:])
        results.append(dict(arm=arm,comparisons=comparisons,document_isolation='passed',
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        del model,buffers,hidden,after,gradient
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    write(args.output,dict(status='passed',cases=results,backend='sdpa',dtype='bf16',
        deterministic_probe=True,reference_repeat_control=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',required=True);p.add_argument('--rows',required=True);p.add_argument('--output',required=True)
    p.add_argument('--arms',nargs='+',choices=('P4-iso-4h','P4-4h'),default=('P4-iso-4h','P4-4h'))
    p.add_argument('--components',nargs='+',choices=('lm','aux','combined'),default=('aux','lm','combined'))
    p.add_argument('--anomaly',action='store_true')
    check(p.parse_args())
