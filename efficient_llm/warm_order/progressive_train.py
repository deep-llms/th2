"""Offline single-GPU trainer with guarded full-state continuation."""
import argparse, contextlib, json, math, os, random, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .model import Config, Model
from .progressive import path_for
from .data import Rows, sha


def write_json(path, value):
    p=Path(path); t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'); os.replace(t,p)


def amp(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else contextlib.nullcontext()


def lr_at(step, total, peak, warmup):
    if step < warmup:return peak*(step+1)/warmup
    fraction=(step-warmup)/max(1,total-warmup-1)
    return peak*(.1+.9*.5*(1+math.cos(math.pi*fraction)))


def source_hashes():
    root=Path(__file__).resolve().parents[2]
    paths=['efficient_llm/warm_order/model.py','efficient_llm/warm_order/data.py','efficient_llm/warm_order/train.py','efficient_llm/warm_order/progressive.py','efficient_llm/warm_order/progressive_train.py','efficient_llm/dot_pool/model.py']
    return {p:sha(root/p) for p in paths}


def save_state(out, model, opt, recipe, step, elapsed, block_tokens, counts):
    p=out/'fullstate.pt';tmp=out/'fullstate.tmp'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),recipe=recipe,step=step,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                    numpy_rng=np.random.get_state(),python_rng=random.getstate(),elapsed=elapsed,block_tokens=block_tokens,counts=counts),tmp)
    os.replace(tmp,p)
    write_json(out/'checkpoint.json',dict(step=step,sha256=sha(p),bytes=p.stat().st_size,block_tokens=block_tokens))


def train(config, out, device='cuda', stop_after=None, resume=False):
    c=dict(config);out=Path(out);path_for(c['arm'],0,c['steps'])
    if device=='cuda' and torch.cuda.device_count()!=1:raise ValueError('Exactly one visible GPU per fit')
    torch.set_num_threads(c.get('threads',4));torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    rows=Rows(c['data'],c['model']['length']); ids=np.load(c['indices'],allow_pickle=False)
    batch=c['batch'];total=c['steps'];needed=batch*total
    if len(ids)<needed or len(np.unique(ids[:needed]))!=needed:raise ValueError('Insufficient unique training rows')
    if ids.min()<0 or ids.max()>=len(rows.ids):raise ValueError('Invalid indices')
    inputs={str(p):sha(p) for p in (Path(c['data']+'.bin'),Path(c['data']+'.starts'),Path(c['indices']))}
    # Dataset files are immutable. Hash at each independent start/resume.
    if c.get('expected_inputs') is not None and inputs!=c['expected_inputs']:raise ValueError('Input identity mismatch')
    recipe=dict(config=c,inputs=inputs,source=source_hashes(),torch=torch.__version__,numpy=np.__version__,precision='bf16' if device.startswith('cuda') else 'fp32')
    model=Model(Config(**c['model'])).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],betas=(.9,.95),weight_decay=.1)
    step=0;elapsed=0.;block_tokens=0;counts={}
    if resume:
        if json.loads((out/'checkpoint.json').read_text())['sha256']!=sha(out/'fullstate.pt'):raise ValueError('Checkpoint hash mismatch')
        state=torch.load(out/'fullstate.pt',map_location='cpu',weights_only=False)
        if state['recipe']!=recipe:raise ValueError('Resume recipe mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);step=state['step']
        torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
        elapsed=state['elapsed'];block_tokens=state['block_tokens'];counts=state['counts']
        if (out/'complete.json').exists():raise ValueError('Already complete')
        write_json(out/'resume.json',dict(resumed_step=step,loaded_fullstate_sha256=sha(out/'fullstate.pt'),source=recipe['source']))
        # Trace may contain updates past the checkpoint following a crash. Refuse ambiguous replay.
        traces=[json.loads(x) for x in (out/'trace.jsonl').read_text().splitlines()] if (out/'trace.jsonl').exists() else []
        if traces and traces[-1]['step']>step:raise ValueError('Trace ahead of checkpoint; explicit recovery required')
    else:
        if out.exists() and any(out.iterdir()):raise ValueError('Nonempty fresh output')
        out.mkdir(parents=True,exist_ok=True);write_json(out/'recipe.json',recipe)
    end=total if stop_after is None else min(total,stop_after)
    if end<=step:raise ValueError('Stop must be later than current step')
    model.train();start=time.monotonic()
    for s in range(step,end):
        ix=ids[s*batch:(s+1)*batch];x=rows.batch(ix,device);path=path_for(c['arm'],s,total)
        opt.zero_grad(set_to_none=True);lr=lr_at(s,total,c['lr'],c['warmup'])
        for group in opt.param_groups:group['lr']=lr
        with amp(device):loss=model.nll(x,path).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        step=s+1;block_tokens+=(2+len(path))*x.numel();key=','.join(map(str,path));counts[key]=counts.get(key,0)+1
        if step% c.get('log_every',50)==0 or step==end:
            if device.startswith('cuda'):torch.cuda.synchronize()
            record=dict(step=step,loss=float(loss),lr=lr,grad_norm=float(norm),path=path,first_row=int(ix[0]),elapsed_seconds=elapsed+time.monotonic()-start,block_tokens=block_tokens)
            with (out/'trace.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if step%c.get('save_every',1000)==0 or step==end:
            save_state(out,model,opt,recipe,step,elapsed+time.monotonic()-start,block_tokens,counts)
    if step==total:
        torch.save(dict(config=c['model'],model=model.state_dict()),out/'weights.pt')
        write_json(out/'complete.json',dict(step=step,arm=c['arm'],seed=c['seed'],input_tokens=needed*c['model']['length'],scored_tokens=needed*(c['model']['length']-c['model']['prefix']),block_tokens=block_tokens,elapsed_seconds=elapsed+time.monotonic()-start,weights_sha256=sha(out/'weights.pt'),fullstate_sha256=sha(out/'fullstate.pt'),counts=counts))
    return model,opt


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    train(json.loads(Path(a.config).read_text()),a.out,a.device,a.stop_after,a.resume)
if __name__=='__main__':main()
