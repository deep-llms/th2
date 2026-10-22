"""Real CUDA BF16 grouped GEMM checks; caller owns GPU allocation."""
import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def relative(a,b):
    return float((a.detach().float()-b.detach().float()).norm()/b.detach().float().norm().clamp_min(1e-12))


def main():
    import torch
    from transformers import Qwen3Config
    from bdot.model import BDOTForCausalLM
    from bdot.routing import RoutingConfig
    from bdot import grouped
    from packed_training import PackedCollator, model_inputs
    from packed_fa4 import register_backend
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--v3',action='store_true')
    args=p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):p.error('Explicit GPU allocation required')
    if args.output.exists():p.error('Fresh result path required')
    torch.set_num_threads(1);torch.manual_seed(31)
    records=[]
    # Irregular token groups, no padding/divisibility assumption, real autograd.
    for sizes in ([1,17,65],[128,256,384]):
        layers=torch.nn.ModuleList([torch.nn.Linear(128,256,bias=False,device='cuda') for _ in range(4)])
        ref=copy.deepcopy(layers)
        x=torch.randn(sum(sizes),128,device='cuda',requires_grad=True)
        y=x.detach().clone().requires_grad_()
        offsets=torch.tensor(sizes,device='cuda',dtype=torch.int32).cumsum(0,dtype=torch.int32)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            a=grouped.linear(x,[layers[i] for i in (0,2,3)],offsets)
            b=torch.cat([ref[i](v) for i,v in zip((0,2,3),y.split(sizes))])
        signal=torch.randn_like(a)
        a.backward(signal);b.backward(signal)
        errors=dict(output=relative(a,b),input_gradient=relative(x.grad,y.grad),
            weight_gradient=max(relative(layers[i].weight.grad,ref[i].weight.grad) for i in (0,2,3)))
        assert max(errors.values()) < .01,errors
        assert layers[1].weight.grad is None
        records.append(dict(case='projection',sizes=sizes,errors=errors))
        print('GROUPED_CASE '+json.dumps(records[-1]),flush=True)
    del layers,ref,x,y,a,b
    rows=[dict(input_ids=torch.randint(2,256,(64,)).tolist(),labels=[0]*64,
               attention_mask=[1]*64,segments=[0]*17+[1]*47) for _ in range(2)]
    for row in rows:row['labels']=row['input_ids'].copy()
    variants = [(True,.01,False,'both'),(True,.3,False,'both'),(False,.01,True,'both'),(False,.3,True,'both')]
    if args.v3:
        variants = [(reuse,1.,True,scope) for reuse in (False,True) for scope in ('both','query','mlp')]
    for backend in ('sdpa','fa4'):
        for reuse,phase,checkpointing,scope in variants:
            c=Qwen3Config(vocab_size=256,hidden_size=256,intermediate_size=768,num_hidden_layers=6,
                num_attention_heads=4,num_key_value_heads=2,head_dim=128,
                attention_dropout=0.,tie_word_embeddings=True,rms_norm_eps=1e-6,max_position_embeddings=128)
            c._attn_implementation='sdpa' if backend=='sdpa' else register_backend()
            c.use_cache=False
            extra = dict(curriculum='none',logit_cap=2.,select_on='probs',placement_film=True,
                         placement_film_scope=scope) if args.v3 else {}
            c.bdot=RoutingConfig(n_stem=1,n_coda=1,S=4,r=2 if args.v3 and not reuse else 1,
                                reuse=reuse,explore_eps_max=0.,**extra).to_dict()
            a=BDOTForCausalLM(c).cuda()
            if args.v3:
                with torch.no_grad():
                    for name,parameter in a.named_parameters():
                        if 'placement_film' in name:parameter.uniform_(-.1,.1)
            # Clear route margins isolate numerical GEMM checks from discrete ties.
            with torch.no_grad():
                for s,step in enumerate(a.routed_steps):
                    step.router.bias.copy_(torch.tensor([.1,.2,.3,.4],device='cuda'))
                    step.router.bias[s]+=2.
            b=copy.deepcopy(a);b.options=replace(b.options,expert_backend='grouped')
            b.config.bdot=b.options.to_dict()
            for step in b.routed_steps:step.options=b.options
            a.progress=b.progress=phase;a.capture_routes=b.capture_routes=True
            if checkpointing:
                for m in (a,b):m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            inputs=model_inputs({k:v.cuda() if isinstance(v,torch.Tensor) else v
                for k,v in PackedCollator(attention_backend=backend)(rows).items()})
            if args.v3:
                # Mixed legal routes exercise occupied off-home PF groups while
                # holding discrete choices identical for the kernel comparison.
                paths=torch.tensor([[0,1,2,3],[1,0,3,2],
                                    [1,0,3,2] if reuse else [2,0,1,3]],device='cuda')
                inputs['bdot_forced_routes']=paths[torch.arange(128,device='cuda')%3].T.reshape(4,2,64)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                oa=a(**inputs);ob=b(**inputs)
            assert torch.equal(a.last_routes,b.last_routes),'Route margin fixture changed'
            oa.loss.backward();ob.loss.backward()
            squared=norm=0.
            for (name,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):
                assert (pa.grad is None)==(pb.grad is None),name
                if pa.grad is not None:
                    assert torch.isfinite(pb.grad).all(),name
                    squared+=float((pa.grad.float()-pb.grad.float()).square().sum())
                    norm+=float(pa.grad.float().square().sum())
            errors=dict(logits=relative(ob.logits,oa.logits),gradient=(squared/max(norm,1e-30))**.5,
                        loss_absolute=abs(float(oa.loss)-float(ob.loss)))
            assert errors['logits']<.02 and errors['gradient']<.05,errors
            # First-step AdamW is discontinuous near zero gradients: BF16
            # rounding can reverse one coordinate's full +/-LR step. Report
            # its maximum and whole-update relative error; do not call it exact.
            before=[pa.detach().clone() for pa in a.parameters()]
            for m in (a,b):torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.1).step()
            delta=max(float((pa-pb).detach().abs().max()) for pa,pb in zip(a.parameters(),b.parameters()))
            update_difference=sum(float((pa-pb).detach().square().sum()) for pa,pb in zip(a.parameters(),b.parameters()))
            update_norm=sum(float((pa-original).detach().square().sum()) for pa,original in zip(a.parameters(),before))
            first_update_relative=(update_difference/max(update_norm,1e-30))**.5
            print('GROUPED_FIRST_ADAMW '+json.dumps(dict(backend=backend,reuse=reuse,phase=phase,
                errors=errors,max_abs=delta,update_relative=first_update_relative)),flush=True)
            assert first_update_relative<.10,('First AdamW RMS update difference',first_update_relative)
            # Independent optimizer-state check away from that discontinuity.
            for m in (a,b):
                optimizer=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.1)
                for parameter,original in zip(m.parameters(),before):
                    with torch.no_grad():parameter.copy_(original)
                    optimizer.state[parameter]=dict(step=torch.tensor(5.),
                        exp_avg=torch.full_like(parameter,.001),
                        exp_avg_sq=torch.full_like(parameter,.0001))
                optimizer.step();m.finish_update()
            mature_delta=max(float((pa-pb).detach().abs().max()) for pa,pb in zip(a.parameters(),b.parameters()))
            assert mature_delta<1e-5,('AdamW existing-state difference',mature_delta)
            assert torch.equal(a.balance_bias,b.balance_bias)
            records.append(dict(case='model',backend=backend,reuse=reuse,phase=phase,
                v3=args.v3,placement_film_scope=scope,
                checkpointing=checkpointing,errors=errors,adamw_first_max_abs=delta,
                adamw_first_update_relative=first_update_relative,adamw_existing_state_max_abs=mature_delta))
            print('GROUPED_CASE '+json.dumps(records[-1]),flush=True)
            del a,b,oa,ob,inputs
    result=dict(status='passed',torch=torch.__version__,gpu=torch.cuda.get_device_name(),cases=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('GROUPED_NUMERICS_PASSED',flush=True)


if __name__=='__main__':main()
