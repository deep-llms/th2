"""Production T1/T2/T7/T9/T11/T12/T13/T14, values saved even on failure."""
import argparse, dataclasses, math, tempfile, time, traceback
from pathlib import Path
import numpy as np
import torch
from efficient_llm.dot_pool.rl import *
from efficient_llm.dot_pool.rl_run import optim, update
from efficient_llm.dot_pool.train import seed

def toy(device):
    seed(981);p=Policy(1,fixed=True).to(device);opt=torch.optim.AdamW([v for v in p.parameters() if v.requires_grad],lr=.01,betas=(.9,.95),weight_decay=0.)
    for step in range(1000):
        previous=torch.zeros(2048,dtype=torch.long,device=device);logs=[];kls=[];paths=[]
        for s in range(8):
            lp=p(torch.zeros(2048,1,device=device),s,previous).log_softmax(-1);pr=lp.exp();a=torch.multinomial(pr,1).squeeze(1)
            logs.append(lp.gather(1,a[:,None]).squeeze(1));kls.append((pr*(lp+math.log(4))).sum(1));paths.append(a);previous=a+1
        paths=torch.stack(paths,1).reshape(256,8,8);rewards=(paths==torch.arange(8,device=device)%4).float().sum(-1)
        loss,_=policy_loss(torch.stack(logs,1).reshape(256,8,8),torch.stack(kls,1).reshape(256,8,8),rewards,paths)
        opt.zero_grad();loss.backward();opt.step()
    prob=1.
    for s in range(8):
        pr=p(torch.zeros(1,1,device=device),s,torch.tensor([0 if s==0 else (s-1)%4+1],device=device)).softmax(-1)
        prob*=float(pr[0,s%4])
    return dict(passed=prob>=.9,loop_probability=prob,updates=1000)

def uniform():
    seed(730);v=torch.randint(4,(1000000,8));counts=torch.stack([torch.bincount(v[:,s],minlength=4) for s in range(8)])
    chi=((counts-250000.)**2/250000.).sum(1);p=[]
    for x in chi.tolist():
        z=x/2;p.append(math.erfc(math.sqrt(z))+2*math.sqrt(z/math.pi)*math.exp(-z))
    return dict(passed=min(p)>.01,counts=counts.tolist(),p_values=p,draws_per_step=1000000)

def checks(arm,data,root,device,backend):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);device=torch.device(device)
    tests={};flags=[];c=Config(stem=0,coda=0,backend=backend)
    with torch.device('meta'):meta=Library(c)
    count=meta.counts();tests['T1']=dict(passed=abs(count/12.59e6-1)<.001,nonembedding=count)
    seed(71);m=Library(c).to(device);p=Policy(c.d).to(device)
    arr=np.load(Path(data)/'train.npy')[:256];x,pos,y=batch(arr[:16],device)
    with torch.no_grad():p.W.normal_(std=.02)
    with torch.no_grad(),autocast(device):
        seed(11);a=m(x,pos,p);z=x.clone();z[torch.arange(len(x),device=device),pos+1]=(z[torch.arange(len(x),device=device),pos+1]+11)%2000
        seed(11);b=m(z,pos,p)
    error=float((a['logits']-b['logits']).abs().max());tests['T2']=dict(passed=error<=1e-4 and torch.equal(a['paths'],b['paths']),max_abs_error=error)
    # T14: RL-only leaves every layer gradient absent; CE leaves router gradients absent.
    m.zero_grad();p.zero_grad();loss,stats=rollout(m,p,arr[:8],device);loss.backward()
    absent=all(v.grad is None for v in m.parameters());present=any(v.grad is not None and bool(v.grad.abs().sum()>0) for v in p.parameters())
    m.zero_grad();p.zero_grad()
    with autocast(device):ce=F.cross_entropy(m(x,pos,p)['logits'],y)
    ce.backward();no_policy=all(v.grad is None for v in p.parameters())
    tests['T14']=dict(passed=absent and present and no_policy,rollout_layer_grad_absent=absent,router_grad_present=present,CE_policy_grad_absent=no_policy)
    # T12 actual small frozen optimization, separate from production FR's before/after hashes.
    m.zero_grad();m.requires_grad_(False);h=tensor_hash(m);_,po=optim(m,p,device,True)
    for i in range(3):update(m,p,None,po,None,arr[:8],device,i+1,active=True,group=8)
    h2=tensor_hash(m);tests['T12']=dict(passed=h==h2,before=h,after=h2);m.requires_grad_(True)
    # Resume includes both optimizers, independent reward stream and CPU/CUDA RNG.
    seed(72);m=Library(c).to(device);p=Policy(c.d).to(device);opt,po=optim(m,p,device);stream=RewardStream(256,71)
    cfg=dict(arm='J-RL',seed=72,model=dataclasses.asdict(c),batch=16,lr=.001,spec_sha256=SPEC_SHA)
    def step(i):return float(update(m,p,opt,po,arr[:16],arr[stream.next(8)],device,i,active=True)['ce'])
    for i in range(1,4):step(i)
    save(root/'T9_state',m,p,opt,po,3,cfg,stream);aa=[step(i) for i in range(4,14)]
    restored,_=load(root/'T9_state',m,p,opt,po,cfg,stream,device);bb=[step(i) for i in range(4,14)]
    err=max(abs(a-b) for a,b in zip(aa,bb));tests['T9']=dict(passed=err<1e-4 and restored==3,max_loss_error=err,expected=aa,actual=bb)
    # One chosen path applies to the entire question: batched grouped vs singleton reference.
    m.eval();seed(3);paths=torch.randint(4,(len(x),8),device=device)
    with torch.no_grad(),autocast(device):
        a=m(x,pos,paths=paths)['logits'];b=torch.cat([m(x[i:i+1],pos[i:i+1],paths=paths[i:i+1])['logits'] for i in range(len(x))])
    err=float((a-b).abs().max());uniform_test=uniform();tests['T13']=dict(uniform_test,grouped_max_abs_error=err,passed=uniform_test['passed'] and torch.allclose(a,b,atol=.08,rtol=.02),grouping_tolerance='BF16 atol.08 rtol.02; CPU exact reference also tested')
    if arm=='J-RL':tests['T11']=toy(device)
    # T7 frozen uniform policy, fixed per-fact routes, same E1b fallback LR protocol.
    attempts=[]
    for lr in [.001,.003]:
        seed(71);m=Library(c).to(device);p=Policy(c.d).to(device) if arm=='J-RL' else None;opt,_=optim(m,None,device)
        for g in opt.param_groups:g['lr']=lr
        paths=(torch.arange(8,device=device)%4).expand(256,8) if arm=='L' else torch.randint(4,(256,8),device=device)
        x7,pos7,y7=batch(arr,device)
        for i in range(300):
            opt.zero_grad()
            with autocast(device):r=m(x7,pos7,p,paths=paths);ce=F.cross_entropy(r['logits'],y7)
            ce.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();value=float(ce)
            if value<.1:break
        attempts.append(dict(lr=lr,steps=i+1,loss=value,passed=value<.1))
        if value<.1:break
    tests['T7']=dict(passed=attempts[-1]['passed'],attempts=attempts,arm=arm)
    write(root/'checks.json',dict(arm=arm,tests=tests,passed=all(v['passed'] for v in tests.values())))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--backend',default='sdpa');a=ap.parse_args()
    checks(a.arm,a.data,a.output,a.device,a.backend)
