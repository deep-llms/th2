"""Allocated-GPU numerical checks for fixed-attention partial MLP routing."""
import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from transformers import Qwen3Config
    from bdot.model import BDOTForCausalLM
    from bdot.routing import RoutingConfig
    from packed_training import PackedCollator,model_inputs
    from packed_fa4 import register_backend
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--followup',action='store_true')
    a=p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or a.output.exists():
        p.error('Require allocated GPU and fresh output')
    torch.set_num_threads(1)
    def rel(x,y):
        return float((x.detach().float()-y.detach().float()).norm()/y.detach().float().norm().clamp_min(1e-12))
    def compare(left,right,ia,ib,case):
        left.zero_grad(set_to_none=True);right.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            x=left(**ia);y=right(**ib)
        x.loss.backward();y.loss.backward()
        numerator=denominator=0.
        for (name,u),(other,v) in zip(left.named_parameters(),right.named_parameters()):
            assert name==other and (u.grad is None)==(v.grad is None),name
            if u.grad is not None:
                assert torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all(),name
                numerator+=float((u.grad.float()-v.grad.float()).square().sum())
                denominator+=float(u.grad.float().square().sum())
        record=dict(case=case,logits_relative=rel(y.logits,x.logits),
                    gradient_relative=(numerator/max(denominator,1e-30))**.5,
                    loss_absolute=abs(float(y.loss)-float(x.loss)))
        assert record['logits_relative']<.02 and record['gradient_relative']<.05,record
        assert torch.equal(left.last_routes,right.last_routes)
        before=[u.detach().clone() for u in left.parameters()]
        for net in (left,right):torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.1).step()
        err=sum(float((u-v).detach().square().sum()) for u,v in zip(left.parameters(),right.parameters()))
        norm=sum(float((u-v).detach().square().sum()) for u,v in zip(left.parameters(),before))
        record['first_adamw_update_relative']=(err/max(norm,1e-30))**.5
        assert record['first_adamw_update_relative']<.1,record
        return record
    torch.manual_seed(31)
    rows=[]
    for _ in range(2):
        ids=torch.randint(2,256,(64,)).tolist()
        rows.append(dict(input_ids=ids,labels=ids.copy(),attention_mask=[1]*64,segments=[0]*17+[1]*47))
    inputs={backend:model_inputs({k:v.cuda() if isinstance(v,torch.Tensor) else v
        for k,v in PackedCollator(attention_backend=backend)(rows).items()}) for backend in ('sdpa','fa4')}
    paths=torch.tensor([[0,1,2,3],[1,0,3,2],[1,1,1,2]],device='cuda')
    forced=paths[torch.arange(128,device='cuda')%3].T.reshape(4,2,64)
    records=[]
    cases=[(f,{}) for f in (.25,.5,1.)]
    if a.followup:
        cases=[(f,dict(router_metric=metric,router_detach_input=detach))
               for f in (.25,.5) for metric,detach in [('linear',True),('cosine',False),('cosine',True)]]
    for fraction,options in cases:
        for backend in ('sdpa','fa4'):
            c=Qwen3Config(vocab_size=256,hidden_size=256,intermediate_size=768,num_hidden_layers=6,
                num_attention_heads=4,num_key_value_heads=2,head_dim=128,attention_dropout=0.,
                tie_word_embeddings=True,rms_norm_eps=1e-6,max_position_embeddings=128,use_cache=False)
            c._attn_implementation='sdpa' if backend=='sdpa' else register_backend()
            c.bdot=RoutingConfig(n_stem=1,n_coda=1,S=4,route_scope='mlp',routed_mlp_fraction=fraction,
                curriculum='A',step_film=False,logit_cap=2.,select_on='probs',**options).to_dict()
            left=BDOTForCausalLM(c).cuda()
            native=left.native_reference().cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                x=left(**inputs[backend]);y=native(**inputs[backend])
            home=dict(case='home',fraction=fraction,backend=backend,router_options=options,logits_relative=rel(x.logits,y.logits),
                      loss_absolute=abs(float(x.loss)-float(y.loss)))
            assert home['logits_relative']<.02 and home['loss_absolute']<.01,home
            records.append(home)
            del native,x,y
            right=copy.deepcopy(left);right.options=replace(right.options,expert_backend='grouped')
            right.config.bdot=right.options.to_dict()
            for step in right.routed_steps:step.options=right.options
            right.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            left.progress=right.progress=1.;left.capture_routes=right.capture_routes=True
            fixture={**inputs[backend],'bdot_forced_routes':forced}
            records.append(compare(left,right,fixture,fixture,dict(kind='grouped-replay',fraction=fraction,backend=backend,**options)))
            print('MLP_GPU_CASE '+json.dumps(records[-1]),flush=True)
            if backend=='sdpa':
                other=copy.deepcopy(left)
                other.set_attn_implementation(register_backend())
                records.append(compare(left,other,fixture,{**inputs['fa4'],'bdot_forced_routes':forced},
                    dict(kind='sdpa-fa4',fraction=fraction,**options)))
                del other
            # Document isolation and future-token causality on the new FA4 path.
            if backend=='fa4':
                right.eval();right.routing_override='fixed'
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    original=right(**inputs['fa4']).logits
                    changed=copy.deepcopy(inputs['fa4']);changed['input_ids'][0,:17]=23
                    isolated=right(**changed).logits
                    torch.testing.assert_close(original[0,17:],isolated[0,17:],atol=1e-5,rtol=1e-5)
                    changed=copy.deepcopy(inputs['fa4']);changed['input_ids'][0,8:]=24
                    causal=right(**changed).logits
                    torch.testing.assert_close(original[0,:8],causal[0,:8],atol=1e-5,rtol=1e-5)
                records.append(dict(case='fa4-isolation-causality',fraction=fraction,status='passed'))
            del left,right
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(status='passed',torch=torch.__version__,cases=records),indent=2))
    print('MLP_GPU_NUMERICS_PASSED',flush=True)


if __name__=='__main__':main()
