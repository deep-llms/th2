"""Teacher-anchored identity/swap controls with exact fullstate resume."""
import argparse,copy,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .model import Config,Model,ORDERS
from .order_distill import objective,ARMS
from .data import Rows,sha
from .train import amp,lr_at,write_json,source_hashes


def update(model,teacher,x,arm,opt):
    opt.zero_grad(set_to_none=True)
    loss,ce,kl=objective(model,teacher,x,arm)
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    return dict(loss=float(loss.detach()),ce=float(ce.detach()),kl=float(kl.detach()),grad_norm=float(norm))


def train(c,out,device='cuda',stop_after=None,resume=False):
    out=Path(out);arm=c['arm'];assert arm in ARMS
    if device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('Exactly one GPU required')
    if not resume and out.exists():raise ValueError('Fresh output required')
    if resume and (out/'complete.json').exists():raise ValueError('Already terminal')
    torch.set_num_threads(c.get('threads',4));torch.manual_seed(c['seed']);random.seed(c['seed']);np.random.seed(c['seed'])
    inputs={p:sha(p) for p in (c['data']+'.bin',c['data']+'.starts',c['indices'],c['initial_weights'])}
    if inputs!=c['expected_inputs']:raise ValueError('Input mismatch')
    root=Path(__file__).resolve().parents[2];sources=source_hashes()|{p:sha(root/p) for p in ('efficient_llm/warm_order/order_distill.py','efficient_llm/warm_order/order_distill_train.py')}
    recipe=dict(config=c,inputs=inputs,sources=sources,torch=torch.__version__,numpy=np.__version__,precision='bf16' if device.startswith('cuda') else 'fp32')
    state=torch.load(c['initial_weights'],map_location='cpu',weights_only=False);base=Model(Config(**state['config']));base.load_state_dict(state['model'])
    if state['config']!=c['model']:raise ValueError('Architecture changed')
    model=base.to(device);teacher=copy.deepcopy(base).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],betas=(.9,.95),weight_decay=.1)
    rows=Rows(c['data'],c['model']['length']);ids=np.load(c['indices']);total=c['steps'];batch=c['batch']
    if len(ids)!=total*batch or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=len(rows.ids):raise ValueError('Training selection mismatch')
    step=0;elapsed=0.
    if resume:
        receipt=json.loads((out/'checkpoint.json').read_text())
        if sha(out/'fullstate.pt')!=receipt['sha256']:raise ValueError('Checkpoint mismatch')
        s=torch.load(out/'fullstate.pt',map_location='cpu',weights_only=False)
        if s['recipe']!=recipe or s['step']!=receipt['step']:raise ValueError('Resume recipe mismatch')
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);step=s['step'];elapsed=s['elapsed']
        torch.set_rng_state(s['torch_rng']);np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
        if s['cuda_rng']:torch.cuda.set_rng_state_all(s['cuda_rng'])
    else:out.mkdir(parents=True);write_json(out/'recipe.json',recipe)
    end=min(total,stop_after) if stop_after is not None else total
    if end<=step:raise ValueError('Resume must advance')
    started=time.monotonic();model.train()
    for i in range(step,end):
        lr=lr_at(i,total,c['lr'],c['warmup'])
        for group in opt.param_groups:group['lr']=lr
        x=rows.batch(ids[i*batch:(i+1)*batch],device)
        with amp(device):record=update(model,teacher,x,arm,opt)
        step=i+1
        if step%c.get('log_every',50)==0 or step==end:
            if device.startswith('cuda'):torch.cuda.synchronize()
            record.update(step=step,lr=lr,elapsed=elapsed+time.monotonic()-started)
            with (out/'trace.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            write_json(out/'progress.json',record)
        if step%c.get('save_every',1000)==0 or step==end:
            s=dict(model=model.state_dict(),optimizer=opt.state_dict(),recipe=recipe,step=step,elapsed=elapsed+time.monotonic()-started,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
            torch.save(s,out/'fullstate.tmp');os.replace(out/'fullstate.tmp',out/'fullstate.pt');write_json(out/'checkpoint.json',dict(step=step,sha256=sha(out/'fullstate.pt')))
    if step==total:
        torch.save(dict(model=model.state_dict(),config=c['model'],arm=arm),out/'weights.pt')
        write_json(out/'complete.json',dict(arm=arm,seed=c['seed'],step=step,input_tokens=total*batch*c['model']['length'],elapsed=elapsed+time.monotonic()-started,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),fullstate_sha256=sha(out/'fullstate.pt'),weights_sha256=sha(out/'weights.pt'),compute_note='Two student six-block forward/backward passes and one frozen six-block teacher pass per update. Full vocabulary CE/KL calculated in every arm. Block work is not measured FLOPs; report latency separately.'))
    if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher gradient')
    return model,opt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args();train(json.loads(Path(a.config).read_text()),a.out,a.device,a.stop_after,a.resume)
