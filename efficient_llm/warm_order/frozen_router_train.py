"""Authenticated router-only training with immutable body and exact resume."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import Rows, sha
from .frozen_router import load_initial
from .soft_exploration_train import amp, lr_at, save_state, write_json


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ['efficient_llm/warm_order/'+n+'.py' for n in
             ('model', 'data', 'soft_route', 'soft_exploration_train',
              'frozen_router', 'frozen_router_train')]
    paths.append('efficient_llm/dot_pool/model.py')
    return {p: sha(root/p) for p in paths}


def train(config, out, device='cuda', stop_after=None, resume=False):
    c = dict(config); out = Path(out)
    if str(device).startswith('cuda') and torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible GPU required')
    if c['schedule'] != 'router-only-warmup-cosine' or c['steps'] <= 0 or not 0 <= c['warmup'] < c['steps']:
        raise ValueError('Invalid router schedule')
    if not resume and out.exists() and any(out.iterdir()):
        raise ValueError('Fresh output required')
    torch.set_num_threads(c.get('threads', 4))
    torch.manual_seed(c['seed']); np.random.seed(c['seed']); random.seed(c['seed'])
    inputs = {p: sha(p) for p in (c['data']+'.bin', c['data']+'.starts', c['indices'])}
    if inputs != c['expected_inputs']: raise ValueError('Changed data/order')
    ids = np.load(c['indices'], allow_pickle=False); needed = c['steps']*c['batch']
    rows = Rows(c['data'], c['model']['length'])
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) != needed or
            len(np.unique(ids)) != needed or ids.min() < 0 or ids.max() >= len(rows.ids)):
        raise ValueError('Invalid recorded row order')
    model = load_initial(c)
    frozen = model.frozen_state()  # CPU snapshot; never retain a second GPU body.
    model.to(device)
    opt = torch.optim.AdamW(model.router_parameters(), lr=c['lr'], betas=(.9,.95), weight_decay=0)
    recipe = dict(config=c, inputs=inputs, source=source_hashes(), torch=torch.__version__,
                  numpy=np.__version__, precision='bf16' if str(device).startswith('cuda') else 'fp32')
    step=0; elapsed=0.; block_tokens=0; counts={}
    if resume:
        meta=json.loads((out/'checkpoint.json').read_text())
        if sha(out/'fullstate.pt')!=meta['sha256']: raise ValueError('Checkpoint hash mismatch')
        state=torch.load(out/'fullstate.pt',map_location='cpu',weights_only=False)
        if state['recipe']!=recipe or state['step']!=meta['step'] or (out/'complete.json').exists():
            raise ValueError('Invalid resume recipe/state')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        step=state['step']; elapsed=state['elapsed']; block_tokens=state['block_tokens']; counts=state['counts']
        trace=[json.loads(x) for x in (out/'trace.jsonl').read_text().splitlines()]
        if trace and trace[-1]['step']>step: raise ValueError('Trace ahead of checkpoint')
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        if str(device).startswith('cuda'):
            if len(state['cuda_rng'])!=1: raise ValueError('Wrong CUDA RNG state')
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        model.assert_frozen(frozen)
        write_json(out/'resume.json',dict(resumed_step=step,loaded_fullstate_sha256=meta['sha256'],source=recipe['source']))
    else:
        out.mkdir(parents=True,exist_ok=True); write_json(out/'recipe.json',recipe)
    end=c['steps'] if stop_after is None else min(stop_after,c['steps'])
    if end<=step: raise ValueError('No remaining updates')
    start=time.monotonic(); model.train()
    for s in range(step,end):
        opt.param_groups[0]['lr']=lr_at(s,c['steps'],c['lr'],c['warmup'])
        x=rows.batch(ids[s*c['batch']:(s+1)*c['batch']],device)
        opt.zero_grad(set_to_none=True)
        with amp(device): loss=model.nll(x).mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite router loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.router_parameters(),1.,error_if_nonfinite=True)
        opt.step(); step=s+1; block_tokens+=18*x.numel()
        for depth, choices in enumerate(model.last_routes):
            for k,n in enumerate(torch.bincount(choices.flatten(),minlength=4).tolist()):
                key=f'{depth}/{k}'; counts[key]=counts.get(key,0)+n
        if step%c.get('log_every',50)==0 or step==end:
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            record=dict(step=step,loss=float(loss.detach()),lr=opt.param_groups[0]['lr'],grad_norm=float(norm),
                        first_row=int(ids[s*c['batch']]),elapsed_seconds=elapsed+time.monotonic()-start,block_tokens=block_tokens)
            with (out/'trace.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            write_json(out/'progress.json',record);print(json.dumps(record),flush=True)
        if step%c.get('save_every',1000)==0 or step==end:
            model.assert_frozen(frozen)
            save_state(out,model,opt,recipe,step,elapsed+time.monotonic()-start,block_tokens,counts)
    if step==c['steps']:
        model.assert_frozen(frozen)
        torch.save(dict(config=c['model'],arm=c['arm'],seed=c['seed'],model=model.state_dict(),
                        canonical_probability=c['canonical_probability'],router_width=c['router_width']),out/'weights.pt')
        write_json(out/'complete.json',dict(step=step,name=c['name'],arm=c['arm'],seed=c['seed'],frozen_body_exact=True,
            trainable_parameters=sum(p.numel() for p in model.router_parameters()),input_tokens=needed*model.c.length,
            scored_tokens=needed*(model.c.length-model.c.prefix),block_tokens=block_tokens,counts=counts,
            elapsed_seconds=elapsed+time.monotonic()-start,weights_sha256=sha(out/'weights.pt'),fullstate_sha256=sha(out/'fullstate.pt')))
    return model,opt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true')
    a=p.parse_args();train(json.loads(Path(a.config).read_text()),a.out,a.device,a.stop_after,a.resume)
