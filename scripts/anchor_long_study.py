"""Prepare authenticated DCLM rows, continue single blocks, then fit paired architectures."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import os

import numpy as np
import torch

from efficient_llm.warm_order.anchor_long import ARMS, END, FORK, WARM, LongAnchor, schedule
from efficient_llm.warm_order.adaptation_runtime import measurement_runtime, training_runtime
from efficient_llm.warm_order.data import Rows, sha
from efficient_llm.warm_order.embedding_anchor_data import BoundRows
from efficient_llm.warm_order.embedding_anchor_eval import CASES
from efficient_llm.warm_order.embedding_anchor_runtime import anchor_runtime
from efficient_llm.warm_order.exposure_recovery import restore_rng
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.model import Config
from scripts.anchor_boundary_study import FIT, MANIFESTS


def read(path):
    return json.loads(path.read_text())


def manifest(seed):
    path = FIT/f's{seed}'/'evaluation/manifest.json'
    if sha(path) != MANIFESTS[seed]:
        raise ValueError('Prior fit manifest changed')
    return read(path)


def prepare(base):
    out = base/'rows'
    out.mkdir(exist_ok=False)
    exclusions = []
    for seed in range(2):
        meta = manifest(seed)
        binding = meta['data_binding']
        data = BoundRows(binding['prior_recipe_pin'],seed=seed)
        if binding != data.binding:
            raise ValueError('Prior data binding mismatch')
        recipe = read(Path(binding['prior_recipe_pin']['path']))['config']
        exclusions.extend([np.load(recipe['indices']), data.calibration_ids, data.measurement_ids])
        parent = meta['checkpoint_proofs']['warmup']
        if sha(parent['path']) != parent['sha256']:
            raise ValueError('Short warm-up checkpoint changed')
    known = Path('resources/layer_adaptation_known_rows_20261121.npz')
    proof = read(Path('results/layer_dependency_20261120/known_exposure_rows.json'))
    if (sha(known) != proof['output_sha256'] or
            binding['input_pins'][recipe['data']+'.bin'] != proof['token_sha256'] or
            binding['input_pins'][recipe['data']+'.starts'] != proof['starts_sha256']):
        raise ValueError('Exclusions do not match stream')
    with np.load(known) as raw:
        exclusions.append(raw['known_e2_rows'])
    pool = np.setdiff1d(data.rows.eligible(),np.unique(np.concatenate(exclusions)),assume_unique=True)
    size = (END-768)*32
    if len(pool) < 2*size:
        raise ValueError(f'Insufficient disjoint rows: have {len(pool)}, require {2*size}')
    selected = np.random.default_rng(20261122).choice(pool,2*size,replace=False).reshape(2,size)
    files = {}
    for seed,values in enumerate(selected):
        path = out/f's{seed}.npy'
        np.save(path,values)
        files[str(seed)] = sha(path)
    write_json(out/'manifest.json',dict(files=files,stem=recipe['data'],eligible_after_exclusions=len(pool),
                                       count_per_seed=size,source=MANIFESTS,document_overlap_unknown=True))


def save_state(path, model, optimizer, seed, arm, step, binding):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,arm=arm,
                    step=step,binding=binding,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np.random.get_state(),python_rng=random.getstate()),tmp)
    os.replace(tmp,path)
    write_json(path.with_suffix('.json'),dict(path=str(path),step=step,sha256=sha(path)))


@torch.inference_mode()
def score(model, rows, ids, cases, device):
    result = {}
    model.eval()
    with measurement_runtime(device), anchor_runtime(device):
        for key,case in cases.items():
            chunks = []
            for start in range(0,len(ids),4):
                tokens = rows.batch(ids[start:start+4],device)
                loss = model.token_losses(tokens,case['path'],skip=case['skip'])
                chunks.append(loss.double().mean(-1).cpu().numpy())
            result[key] = np.concatenate(chunks)
            if not np.isfinite(result[key]).all():
                raise ValueError('Nonfinite evaluation')
    return result


def worker(base,seed,arm,qualify=False):
    device = 'cuda'
    meta = manifest(seed)
    config = Config(**meta['checkpoint_proofs']['model_config'])
    row_meta = read(base/'rows/manifest.json')
    row_file = base/'rows'/f's{seed}.npy'
    if sha(row_file) != row_meta['files'][str(seed)]:
        raise ValueError('Reserved rows changed')
    ids = np.load(row_file,mmap_mode='r')
    rows = Rows(row_meta['stem'])
    bound = meta['data_binding']; recipe = read(Path(bound['prior_recipe_pin']['path']))['config']
    cal_file = recipe['calibration_indices']
    if sha(cal_file) != bound['input_pins'][cal_file]:
        raise ValueError('Calibration rows changed')
    cal = np.load(cal_file)
    out = base/f's{seed}'/(f'qualification-{arm}' if qualify else arm)
    out.mkdir(parents=True,exist_ok=False)
    warm = meta['checkpoint_proofs']['warmup']
    parent = warm if arm=='shared' or qualify else read(base/f's{seed}'/'shared/fork.json')
    if sha(parent['path']) != parent['sha256']:
        raise ValueError('Parent state changed')
    state = torch.load(parent['path'],weights_only=False,map_location='cpu')
    step = state['step']
    if step != (768 if arm=='shared' or qualify else FORK) or state['arm']!='shared':
        raise ValueError('Wrong ancestry')
    model = LongAnchor(config,arm).to(device)
    weights = dict(state['model'])
    if arm=='reader':
        weights['query'] = torch.zeros_like(model.query)
    model.load_state_dict(weights)
    body = [p for name,p in model.named_parameters() if name!='query']
    opt = torch.optim.AdamW(body,lr=.0006,betas=(.9,.95),eps=1e-8,weight_decay=0)
    opt.load_state_dict(state['optimizer'])
    if arm=='reader':
        opt.add_param_group(dict(params=[model.query]))
    restore_rng(state,device)
    del state
    binding = dict(parent_sha256=parent['sha256'],rows_sha256=sha(row_file),arm=arm,seed=seed)
    single = {k:v for k,v in CASES.items() if k.startswith('single')}
    def update(index, force_stack=False):
        exp = schedule(index)
        path = (0,1,2,3) if force_stack else exp.path
        tokens = rows.batch(ids[exp.row_start:exp.row_stop],device)
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device,dtype=torch.bfloat16):
            loss = model.nll(tokens,path,donor=None if force_stack else exp.donor).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        return float(loss.detach())
    if qualify:
        # Compare independent native CPU and GPU reference on an actual authenticated row.
        from efficient_llm.warm_order.embedding_anchor import EmbeddingAnchor
        cpu = EmbeddingAnchor(config).eval()
        cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items() if k!='query'})
        x = rows.batch(cal[:1],'cpu')
        with torch.inference_mode(), measurement_runtime('cpu'), anchor_runtime('cpu'):
            h = cpu.hidden(x,(0,1,2,3),alpha=1.)
            if arm in ('additive','reader'):
                e = cpu.embed(x); h=e; pos=torch.arange(x.shape[1]).expand_as(x); seg=torch.zeros_like(x)
                for idx,block in enumerate(cpu.blocks):
                    h=block(h if idx==0 else h+.5*e,pos,seg)
            ref=cpu.readout_hidden(h)[:,config.prefix-1:-1]
            target=x[:,config.prefix:]
            ref_nll=torch.nn.functional.cross_entropy((ref @ cpu.embed.weight.T).flatten(0,1),target.flatten(),reduction='none')
        with torch.inference_mode(),measurement_runtime(device),anchor_runtime(device):
            y=x.to(device)
            pred=model.hidden(y,(0,1,2,3))
            torch.testing.assert_close(pred.cpu(),h,atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(model.token_losses(y,(0,1,2,3)).cpu().flatten(),ref_nll,atol=1e-4,rtol=1e-4)
        del cpu
        with training_runtime(device),anchor_runtime(device):
            update(step, True)
            save_state(out/'probe.pt',model,opt,seed,arm,step+1,binding)
            first = update(step+1,True)
            def digest():
                hasher=hashlib.sha256()
                for p in model.parameters():
                    for tensor in (p, *opt.state[p].values()):
                        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                return hasher.hexdigest()
            expected = digest()
            saved=torch.load(out/'probe.pt',weights_only=False,map_location='cpu')
            model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); restore_rng(saved,device)
            del saved
            if update(step+1,True)!=first or digest()!=expected:
                raise ValueError('GPU fork/resume parity failure')
        (out/'probe.pt').unlink()
        write_json(out/'passed.json',dict(arm=arm,seed=seed,fp32_parity=True,exact_resume=True))
        return
    end = FORK if arm=='shared' else END
    write_json(out/'recipe.json',dict(binding=binding,start=step,end=end))
    def track(step):
        cases = single if step<=WARM else {'path0123':CASES['path0123'],'path0213':CASES['path0213']}
        if arm=='reader':
            model.gate_moments=torch.zeros(4,3,dtype=torch.float64,device=device)
        metrics=score(model,rows,cal,cases,device)
        gate={}
        if model.gate_moments is not None:
            for i,(total,squares,count) in enumerate(model.gate_moments.tolist()):
                if count:
                    mean=total/count
                    gate[str(i)]=dict(mean=mean,sd=max(0.,squares/count-mean**2)**.5,count=int(count))
            model.gate_moments=None
        with (out/'calibration.jsonl').open('a') as f:
            print(json.dumps(dict(step=step,gate=gate,nll={k:float(v.mean()) for k,v in metrics.items()})),file=f)
    track(step)
    with training_runtime(device),anchor_runtime(device):
        while step<end:
            loss = update(step)
            step+=1
            if step%128==0:
                write_json(out/'progress.json',dict(step=step,end=end,last_batch_nll=loss))
            if step%1024==0:
                track(step)
            if step%4096==0 and step<end:
                save_state(out/'latest.pt',model,opt,seed,arm,step,binding)
    save_state(out/('fork.pt' if arm=='shared' else 'terminal.pt'),model,opt,seed,arm,step,binding)
    if arm!='shared':
        measurement=np.array(meta['row_ids'],dtype=np.int64)
        values=score(model,rows,measurement,CASES,device)
        np.savez_compressed(out/'scores.npz',row_ids=measurement,**values)
        write_json(out/'summary.json',dict(seed=seed,arm=arm,rows=len(measurement),
                                          nll={k:float(v.mean()) for k,v in values.items()}))
    write_json(out/'complete.json',dict(step=step,binding=binding))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare','qualify','fit'])
    p.add_argument('--base',required=True,type=Path)
    p.add_argument('--seed',type=int,choices=(0,1))
    p.add_argument('--arm',choices=('shared',*ARMS))
    args=p.parse_args()
    if args.action=='prepare':
        prepare(args.base)
    else:
        worker(args.base,args.seed,args.arm,args.action=='qualify')
