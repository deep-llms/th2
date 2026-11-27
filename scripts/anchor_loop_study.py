"""Paired serial and learned-memory recurrence from the verified long warm-up fork."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from efficient_llm.warm_order.adaptation_runtime import measurement_runtime,training_runtime
from efficient_llm.warm_order.anchor_long import END,FORK,schedule
from efficient_llm.warm_order.anchor_loop import LoopAnchor
from efficient_llm.warm_order.data import Rows,sha
from efficient_llm.warm_order.embedding_anchor_data import BoundRows
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.exposure_recovery import restore_rng
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.model import Config
from scripts.anchor_long_study import manifest,read,save_state,score
from scripts.anchor_long_queue import stage,supervise

OLD=Path('/mnt/local/_outputs/deep-llms_th2/anchor_long_20261122_a01')
ROW_PINS=('70fc5792ed95b9ad3931a9604f29670d6769905407c06eb24a62cf8df0fe2853',
          '46f6f3bd65b08297396799b458c91ac5c034293a364244d4e9fb02acf122e0b9')
BACKUP='a5539c41e2a67212150746604c33aa3f8099f3b93119c6d95364ccb2f29ce0ec'
HANDOFF='1d4948144da0e3e9afd38d398af68da445bcabfb081a69e5541c351b32712bd7'
NATIVE=(0,1,2,3)
SWAP=(0,2,1,3)


def parent(seed):
    path=OLD/'backup'/f's{seed}'/'shared/fork.pt'
    if sha(OLD/'backup.json')!=BACKUP:
        raise ValueError('Prior backup manifest mismatch')
    return path,read(OLD/'backup.json')[str(path)]


def prepare(base):
    for seed in (0,1):
        meta=manifest(seed)
        data=BoundRows(meta['data_binding']['prior_recipe_pin'],seed=seed)
        if data.binding!=meta['data_binding'] or sha(OLD/'rows'/f's{seed}.npy')!=ROW_PINS[seed]:
            raise ValueError('Data changed')
        path,digest=parent(seed)
        if sha(path)!=digest:
            raise ValueError('Original shared fork changed')
    write_json(base/'prepared.json',dict(rows=ROW_PINS,backup=BACKUP))


def digest(model,opt):
    h=hashlib.sha256()
    for p in model.parameters():
        for v in (p,*opt.state[p].values()):
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def worker(base,seed,arm,qualify):
    meta=manifest(seed);config=Config(**meta['checkpoint_proofs']['model_config'])
    bind=meta['data_binding']; rp=bind['prior_recipe_pin']
    if sha(rp['path'])!=rp['sha256']:
        raise ValueError('Historical recipe mismatch')
    recipe=read(Path(rp['path']))['config']
    row_path=OLD/'rows'/f's{seed}.npy'
    if sha(row_path)!=ROW_PINS[seed] or sha(recipe['calibration_indices'])!=bind['input_pins'][recipe['calibration_indices']]:
        raise ValueError('Row binding mismatch')
    rows=Rows(recipe['data']);ids=np.load(row_path,mmap_mode='r')
    cal=np.load(recipe['calibration_indices'])
    path,pin=parent(seed)
    if sha(path)!=pin:
        raise ValueError('Parent checkpoint changed')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['step']!=FORK or state['seed']!=seed or state['arm']!='shared' or state['binding']['rows_sha256']!=ROW_PINS[seed]:
        raise ValueError('Invalid fork')
    model=LoopAnchor(config,arm).cuda()
    weights=dict(state['model'])
    memory={k:p for k,p in model.named_parameters() if k.startswith('memory_')}
    model.load_state_dict(weights|{k:torch.zeros_like(p) for k,p in memory.items()})
    opt=torch.optim.AdamW([p for k,p in model.named_parameters() if k not in memory],
                          lr=.0006,betas=(.9,.95),eps=1e-8,weight_decay=0)
    opt.load_state_dict(state['optimizer'])
    if memory:opt.add_param_group(dict(params=list(memory.values())))
    restore_rng(state,'cuda')
    del weights,state
    out=base/f's{seed}'/(f'qualification-{arm}' if qualify else arm)
    out.mkdir(parents=True,exist_ok=False)
    binding=dict(parent_sha256=pin,rows_sha256=ROW_PINS[seed],seed=seed,arm=arm)

    def update(step):
        e=schedule(step)
        x=rows.batch(ids[e.row_start:e.row_stop],'cuda')
        opt.zero_grad(set_to_none=True);model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model.nll(x,NATIVE).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        return float(loss.detach())

    if qualify:
        cpu=LoopAnchor(config,arm).eval()
        for test_gain in ((0.,.1) if memory else (0.,)):
            with torch.no_grad():
                if memory:model.memory_gain.fill_(test_gain)
            cpu.load_state_dict(model.state_dict())
            x=rows.batch(cal[:1],'cpu')
            with torch.inference_mode(),measurement_runtime('cpu'),anchor_runtime('cpu'):
                ref=cpu.hidden(x,NATIVE)
                nll=cpu.token_losses(x,NATIVE)
            with torch.inference_mode(),measurement_runtime('cuda'),anchor_runtime('cuda'):
                y=x.cuda()
                torch.testing.assert_close(model.hidden(y,NATIVE).cpu(),ref,atol=1e-4,rtol=1e-4)
                torch.testing.assert_close(model.token_losses(y,NATIVE).cpu(),nll,atol=1e-4,rtol=1e-4)
        del cpu
        with torch.no_grad():
            if memory:model.memory_gain.zero_()
        with training_runtime('cuda'),anchor_runtime('cuda'):
            update(FORK)
            save_state(out/'probe.pt',model,opt,seed,arm,FORK+1,binding)
            loss=update(FORK+1);full=digest(model,opt)
            saved=torch.load(out/'probe.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved,'cuda')
            del saved
            if update(FORK+1)!=loss or digest(model,opt)!=full:
                raise ValueError('Exact continuation failed')
        (out/'probe.pt').unlink()
        write_json(out/'passed.json',dict(seed=seed,arm=arm,actual_gpu=True,exact_resume=True))
        return
    def eval_path(data,path=NATIVE):
        return score(model,rows,data,{'cost':dict(path=path,skip=None)},'cuda')['cost']
    def calibration(step):
        value=eval_path(cal)
        with (out/'calibration.jsonl').open('a') as f:
            print(json.dumps(dict(step=step,nll=float(value.mean()))),file=f)
    calibration(FORK)
    with training_runtime('cuda'),anchor_runtime('cuda'):
        for step in range(FORK,END):
            loss=update(step);done=step+1
            if done%128==0:write_json(out/'progress.json',dict(step=done,last_batch=loss))
            if done%1024==0:calibration(done)
            if done%4096==0 and done<END:save_state(out/'latest.pt',model,opt,seed,arm,done,binding)
    save_state(out/'terminal.pt',model,opt,seed,arm,END,binding)
    measurement=np.array(meta['row_ids'],dtype=np.int64)
    values={}
    for passes in (1,2,3):
        model.passes=passes
        for name,order in [('native',NATIVE),('swap',SWAP)]:
            values[f'{name}_{passes}']=eval_path(measurement,order)
    if memory:
        model.memory_enabled=False
        values['memory_off_3']=eval_path(measurement)
    np.savez_compressed(out/'scores.npz',row_ids=measurement,**values)
    write_json(out/'summary.json',dict(binding=binding,step=END,rows=len(measurement),score_sha256=sha(out/'scores.npz'),
             nll={k:float(v.mean()) for k,v in values.items()}))


def experiment(base, arms=('serial','memory')):
    jobs=[(seed,arm) for seed in (0,1) for arm in arms]
    module='scripts.anchor_loop_study'
    stage(base,'qualify',jobs,module)
    for seed,arm in jobs:
        gate=read(base/f's{seed}'/f'qualification-{arm}'/'passed.json')
        if gate!=dict(seed=seed,arm=arm,actual_gpu=True,exact_resume=True):
            raise ValueError('Missing actual parity/resume')
    stage(base,'fit',jobs,module)
    report={};backups={}
    for seed,arm in jobs:
        out=base/f's{seed}'/arm
        result=read(out/'summary.json')
        if result['step']!=END or result['rows']!=512 or sha(out/'scores.npz')!=result['score_sha256']:
            raise ValueError('Invalid terminal')
        report[f's{seed}/{arm}']=result
        backup=base/'backup'/f's{seed}'/arm/'terminal.pt'
        backup.parent.mkdir(parents=True);shutil.copyfile(out/'terminal.pt',backup)
        expected=read(out/'terminal.json')['sha256']
        if sha(backup)!=expected or sha(out/'terminal.pt')!=expected:
            raise ValueError('Terminal backup mismatch')
        backups[str(backup)]=expected
    write_json(base/'report.json',report);write_json(base/'backup.json',backups)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('queue','static-queue','qualify','fit'))
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=(0,1));p.add_argument('--arm',choices=('serial','memory','static'))
    a=p.parse_args()
    if a.action in ('queue','static-queue'):
        try:
            if a.action == 'static-queue':
                predecessor = OLD.parent/'anchor_loop_20261123_a01/native-verified.json'
                supervise(a.base, prepare, lambda base: experiment(base, ('static',)), predecessor,
                          'cbcc4c70b3bd308abd5ef13321343e450114aeb41d6afa5224b51d14de3930da')
            else:
                supervise(a.base,prepare,experiment,OLD/'native-verified.json',HANDOFF)
        except Exception as err:
            write_json(a.base/'failure.json',dict(reason=repr(err)));raise
    else:worker(a.base,a.seed,a.arm,a.action=='qualify')
