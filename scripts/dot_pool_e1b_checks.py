"""T1/T2/T9 on the actual no-stem/coda production configuration, per arm."""
import argparse, dataclasses, json
from pathlib import Path
import numpy as np
import torch
from efficient_llm.dot_pool.model import Config, Model
from efficient_llm.dot_pool.data import e1_batch, write
from efficient_llm.dot_pool.train import seed, autocast, save, load
from efficient_llm.dot_pool import SPEC_SHA

EXPECTED={'D':(25.18,25.18),'L':(12.59,25.18),'SO':(82.06,25.44),'DOT':(82.10,25.47)}
def config(arm, **kwargs):
    return Config.production('E1',arm,stem=0,coda=0,**kwargs)

def counts(arm):
    with torch.device('meta'): m=Model(config(arm))
    result=m.counts(); total,active=EXPECTED[arm]
    if abs(result['nonembedding_total']/1e6/total-1)>.001 or abs(result['nonembedding_active']/1e6/active-1)>.001:
        raise ValueError('T1 parameter mismatch')
    return result

def checks(arm,data,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    c=config(arm,backend='fa4',expert_kernel='grouped');device=torch.device('cuda');seed(777)
    m=Model(c).to(device); records={'T1':dict(passed=True,counts=counts(arm))}
    # Factual inputs and a separate packed document-isolation probe.
    m.eval();x=torch.tensor([[1,2000,2001,2,3,4],[5,2002,2003,6,7,8]],device=device)
    seg=torch.tensor([[0,0,0,1,1,1]],device=device).expand_as(x);pos=torch.tensor([[0,1,2,0,1,2]],device=device).expand_as(x)
    with torch.no_grad(),autocast(device):
        y=m(x,segments=seg,positions=pos);z=x.clone();z[:,2]+=1;yy=m(z,segments=seg,positions=pos)
    error=max(float((y[:,:2]-yy[:,:2]).abs().max()),float((y[:,3:]-yy[:,3:]).abs().max()))
    records['T2']=dict(passed=error<=1e-4,max_abs_error=error)
    del y,yy;m.train()
    opt=torch.optim.AdamW(m.parameter_groups(),lr=.001,betas=(.9,.95),eps=1e-8,fused=True)
    inp=e1_batch(np.load(Path(data)/'train.npy')[:16],device);targets=16
    def step():
        opt.zero_grad(set_to_none=True);m.gradient_factor=targets
        with autocast(device):out=m(**inp);loss=out['nll_sum']/targets+out['balance_loss']+out['taylor_loss']
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();m.finish_update();m.promote_labels()
        return float(out['nll_sum'].detach()/targets)
    for _ in range(3):step()
    cfg=dict(study='E1',arm=arm,seed=777,lr=.001,batch=16,microbatch=16,steps=13,model=dataclasses.asdict(c),data=str(data),spec_sha256=SPEC_SHA)
    save(root/'resume_state',m,opt,3,cfg,0,1)
    expected=[step() for _ in range(10)]
    restored=load(root/'resume_state',m,opt,dict(cfg,steps=23),0,1,device)
    actual=[step() for _ in range(10)]
    error=max(abs(a-b) for a,b in zip(actual,expected))
    records['T9']=dict(passed=error<=1e-4 and restored==3,max_loss_error=error,expected=expected,actual=actual,restored_step=restored)
    write(root/'checks.json',dict(arm=arm,config=dataclasses.asdict(c),tests=records,passed=all(v['passed'] for v in records.values())))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=list(EXPECTED));p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();checks(a.arm,a.data,a.output)
