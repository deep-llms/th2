"""C6/F7/J7 fixed-token fits; recipe-checked fullstate resumes, no remote actions."""
import argparse,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .model import Config,Model,ORDERS
from .refinement import Refinement
from .data import Rows,sha
from .train import amp,lr_at,write_json,source_hashes


def loss_hidden(h,weight,x,prefix):
    h=h[:,prefix-1:-1].reshape(-1,h.shape[-1]);y=x[:,prefix:].reshape(-1)
    def ce(a,w,b):return F.cross_entropy(F.linear(a,w).float(),b,reduction='none')
    pieces=[]
    for i in range(0,len(y),1024):
        args=(h[i:i+1024],weight,y[i:i+1024])
        pieces.append(checkpoint(ce,*args,use_reentrant=False) if torch.is_grad_enabled() else ce(*args))
    return torch.cat(pieces).reshape(len(x),-1).mean(1)


def update(model,x,arm,opt):
    opt.zero_grad(set_to_none=True)
    if arm=='C6':
        loss=model.nll(x,ORDERS[0]).mean();base_loss=loss;refined_loss=loss
    else:
        a,b=model.states(x);refined_loss=loss_hidden(b,model.base.embed.weight,x,model.base.c.prefix).mean()
        if arm=='J7':base_loss=loss_hidden(a,model.base.embed.weight,x,model.base.c.prefix).mean();loss=(base_loss+refined_loss)*.5
        else:base_loss=refined_loss.detach()*0;loss=refined_loss
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
    return dict(loss=float(loss.detach()),base_loss=float(base_loss.detach()) if arm!='F7' else None,refined_loss=float(refined_loss.detach()),grad_norm=float(norm))


def train(c,out,device='cuda',stop_after=None,resume=False):
    out=Path(out);arm=c['arm'];assert arm in ('C6','F7','J7')
    if device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('Exactly one GPU required')
    if not resume and out.exists():raise ValueError('Fresh output required')
    if resume and (out/'complete.json').exists():raise ValueError('Already terminal')
    torch.set_num_threads(c.get('threads',4));torch.manual_seed(c['seed']);random.seed(c['seed']);np.random.seed(c['seed'])
    inputs={p:sha(p) for p in (c['data']+'.bin',c['data']+'.starts',c['indices'],c['initial_weights'])}
    if inputs!=c['expected_inputs']:raise ValueError('Input mismatch')
    root=Path(__file__).resolve().parents[2];sources=source_hashes()|{p:sha(root/p) for p in ('efficient_llm/warm_order/refinement.py','efficient_llm/warm_order/refinement_train.py')}
    recipe=dict(config=c,inputs=inputs,sources=sources,torch=torch.__version__,numpy=np.__version__,precision='bf16' if device.startswith('cuda') else 'fp32')
    state=torch.load(c['initial_weights'],map_location='cpu',weights_only=False);base=Model(Config(**state['config']));base.load_state_dict(state['model'])
    if state['config']!=c['model']:raise ValueError('Architecture changed')
    model=(base if arm=='C6' else Refinement(base,frozen=arm=='F7')).to(device)
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
        with amp(device):record=update(model,x,arm,opt)
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
        write_json(out/'complete.json',dict(arm=arm,seed=c['seed'],step=step,input_tokens=total*batch*c['model']['length'],elapsed=elapsed+time.monotonic()-started,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),fullstate_sha256=sha(out/'fullstate.pt'),weights_sha256=sha(out/'weights.pt'),compute_note='C6:6 forward+6 backward blocks; F7:7 forward+1 backward; J7:7 forward+7 backward plus two CE exits. Backward FLOPs are not assumed equal to forward. Equal tokens not equal compute.'))
    return model,opt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args();train(json.loads(Path(a.config).read_text()),a.out,a.device,a.stop_after,a.resume)
