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
    return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))


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
    del layers,ref,x,y,a,b
    rows=[dict(input_ids=torch.randint(2,256,(64,)).tolist(),labels=[0]*64,
               attention_mask=[1]*64,segments=[0]*17+[1]*47) for _ in range(2)]
    for row in rows:row['labels']=row['input_ids'].copy()
    for backend in ('sdpa','fa4'):
        for reuse,phase,checkpointing in [(True,.01,False),(True,.3,False),(False,.01,True),(False,.3,True)]:
            c=Qwen3Config(vocab_size=256,hidden_size=256,intermediate_size=768,num_hidden_layers=6,
                num_attention_heads=4,num_key_value_heads=2,head_dim=128,
                attention_dropout=0.,tie_word_embeddings=True,rms_norm_eps=1e-6,max_position_embeddings=128)
            c._attn_implementation='sdpa' if backend=='sdpa' else register_backend()
            c.use_cache=False
            c.bdot=RoutingConfig(n_stem=1,n_coda=1,S=4,r=1,reuse=reuse,explore_eps_max=0.).to_dict()
            a=BDOTForCausalLM(c).cuda()
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
            for m in (a,b):torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.1).step();m.finish_update()
            delta=max(float((pa-pb).abs().max()) for pa,pb in zip(a.parameters(),b.parameters()))
            assert delta<1e-4,('AdamW parameter difference',delta)
            assert torch.equal(a.balance_bias,b.balance_bias)
            records.append(dict(case='model',backend=backend,reuse=reuse,phase=phase,
                checkpointing=checkpointing,errors=errors,adamw_max_abs=delta))
            print('GROUPED_CASE '+json.dumps(records[-1]),flush=True)
            del a,b,oa,ob,inputs
    result=dict(status='passed',torch=torch.__version__,gpu=torch.cuda.get_device_name(),cases=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('GROUPED_NUMERICS_PASSED',flush=True)


if __name__=='__main__':main()
