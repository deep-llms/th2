"""Registered pilot fit around the unchanged, separately qualified update kernel.

Node orchestration binds current ownership, full source, CPU preparation
and successful qualification before calling this entry point.
"""
import gc
import json
import time
from pathlib import Path

from .adaptation_runtime import training_runtime,measurement_runtime
from .data import sha
from .embedding_anchor import exposure
from .embedding_anchor_device import DeviceTrainer,load_prepared,device_sources
from .embedding_anchor_device_eval import evaluate
from .embedding_anchor_prepare import validate_plan
from .embedding_anchor_runtime import anchor_runtime


def validate_gate(gate_path,gate_sha,meta,prepared_sha,device):
    gate_path=Path(gate_path)
    if sha(gate_path)!=gate_sha:raise ValueError('Qualification receipt hash mismatch')
    g=json.loads(gate_path.read_text());fixture=meta['fixture_only'];steps=2 if fixture else 40
    if (g['fixture_only']!=fixture or g['device']!=device or g['source']!=device_sources()
            or g['seed']!=meta['seed'] or g['prepared_sha256']!=prepared_sha
            or g['production_updates_executed']!=0 or g['production_fit_admitted']
            or g['actual_updates_executed']!=steps*11 or not g['lineage_matches_single_segment']
            or set(g['segments'])!={'single','donor-A','donor-C','stack-A','stack-C'}):
        raise ValueError('Qualification does not bind this fit')
    if not fixture and device!='cuda':raise ValueError('Production requires actual CUDA qualification')
    if sha(gate_path.parent/'numerical.json')!=g['numerical_sha256']:
        raise ValueError('Numerical receipt changed')
    numerical=json.loads((gate_path.parent/'numerical.json').read_text())
    with measurement_runtime(device) as policy:
        if numerical['policy']!=policy:raise ValueError('Numerical policy changed')
    for value in g['segments'].values():
        if not value['exact_resume'] or value['updates_per_lineage']!=steps:
            raise ValueError('Missing segment resume proof')
        for key in ('whole','resumed'):
            pin=value[key]
            if sha(pin['path'])!=pin['sha256']:raise ValueError('Qualification state changed')
    return g


def advance_registered(t,end,snapshots):
    """Called only after fit registration and gate validation in run_seed."""
    cap=768 if t.arm=='shared' else 1920
    if type(end) is not int or not t.step<end<=cap:raise ValueError('Invalid registered endpoint')
    t.validate();t.model.train()
    with training_runtime(t.device),anchor_runtime(t.device):
        for step in range(t.step,end):
            e=exposure(step,'C' if t.arm=='shared' else t.arm)
            row=t.update(e,t.row_ids[e.row_start:e.row_stop])
            t.step=step+1;t.trace.append(dict(step=t.step,**row))
            if t.step%32==0:
                progress=dict(arm=t.arm,step=t.step,loss=row['loss'],utc=time.time())
                temp=snapshots/'progress.tmp';temp.write_text(json.dumps(progress)+'\n');temp.replace(snapshots/'progress.json')
                print('ANCHOR_FIT_PROGRESS',json.dumps(progress),flush=True)
            if t.step in (768,1152,1344,1536,1920):
                path=snapshots/f'{t.step:06d}.pt';t.save(path)
    t.validate()


def run_seed(plan_path,plan_sha,prepared_root,prepared_sha,gate_path,gate_sha,seed,out,*,device='cuda'):
    started=time.monotonic()
    if sha(plan_path)!=plan_sha or Path(out).exists():raise ValueError('Registered plan/fresh output required')
    plan=json.loads(Path(plan_path).read_text());prepared_root=Path(prepared_root)
    if sha(prepared_root/'prepared.json')!=prepared_sha:raise ValueError('Preparation hash mismatch')
    prepared=json.loads((prepared_root/'prepared.json').read_text())
    config=validate_plan(plan,fixture_only=prepared['fixture_only'])
    if prepared['plan_sha256']!=plan_sha or seed not in (0,1):raise ValueError('Prepared plan/seed mismatch')
    root=prepared_root/f's{seed}';meta_pin=prepared['seeds'][str(seed)]['config_sha256']
    t,meta=load_prepared(root,meta_pin,device=device)
    if meta['seed']!=seed or t.model.c!=config:raise ValueError('Prepared model/seed mismatch')
    gate=validate_gate(gate_path,gate_sha,meta,prepared_sha,device)
    if gate['config_sha256']!=meta_pin:raise ValueError('Gate prepared config mismatch')
    out=Path(out);out.mkdir(parents=True);shared=out/'shared';shared.mkdir()
    registration=dict(plan_sha256=plan_sha,prepared_sha256=prepared_sha,gate_sha256=gate_sha,
                      fit_source_sha256=sha(__file__),kernel_source=device_sources(),seed=seed,
                      initialization='original CPU step0; qualification updates discarded',
                      physical_fit_updates=3072,fixture_only=meta['fixture_only'])
    (out/'fit-recipe.json').write_text(json.dumps(registration,indent=2)+'\n')
    t.save(shared/'000000.pt');advance_registered(t,768,shared)
    warm=dict(path=str(shared/'000768.pt'),sha256=sha(shared/'000768.pt'))
    data=t.data;del t;gc.collect();runs={}
    prepared_pin=dict(path=str(root),sha256=meta_pin)
    for arm in ('A','C'):
        t=DeviceTrainer.restore(warm['path'],warm['sha256'],config,data,seed=seed,device=device,
                                fork_arm=arm,prepared=prepared_pin)
        directory=out/arm;directory.mkdir();advance_registered(t,1920,directory)
        runs[arm]=dict(path=str(directory/'001920.pt'),sha256=sha(directory/'001920.pt'))
        (directory/'trace.json').write_text(json.dumps(t.trace)+'\n')
        del t;gc.collect()
    evaluate(runs,warm,config,data,out/'evaluation',seed=seed,device=device,prepared=prepared_pin,
             batch=32 if meta['fixture_only'] else 2)
    result=dict(registration=registration,warmup=warm,runs=runs,
                report_sha256=sha(out/'evaluation/REPORT.json'),fit_recipe_sha256=sha(out/'fit-recipe.json'),
                elapsed_seconds=time.monotonic()-started,
                physical_training_input_tokens=3072*32*config.length,
                training_block_forward_minibatch_calls=8448,
                compute_note='Block calls include detached donors; excludes evaluation and qualification, not FLOPs')
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    import argparse
    import torch
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True)
    p.add_argument('--sha',required=True);p.add_argument('--seed',type=int,choices=(0,1),required=True)
    p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4)
    if sha(a.registration)!=a.sha:raise ValueError('Fit registration changed')
    r=json.loads(Path(a.registration).read_text());q=r['gates'][str(a.seed)]
    result=run_seed(r['pilot_plan'],r['pilot_plan_sha256'],r['prepared_root'],r['prepared_sha256'],
                    q['path'],q['sha256'],a.seed,a.out,device='cuda')
    print('ANCHOR_FIT_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
