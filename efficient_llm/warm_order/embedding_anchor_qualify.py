"""Actual-input bounded CUDA qualification; no full architecture fit launch."""
import argparse
import copy
import gc
import json
from pathlib import Path

import torch

from .adaptation_gpu_probe import exact
from .data import sha
from .embedding_anchor_device import DeviceTrainer,load_prepared,check_references,device_sources
from .embedding_anchor_segments import SEGMENTS,run
from .exposure_recovery import restore_rng


def qualify_seed(base,prepared_sha,seed,out,*,device='cuda',fixture_steps=None):
    base,out=Path(base),Path(out)
    if sha(base/'prepared.json')!=prepared_sha or out.exists() or seed not in (0,1):
        raise ValueError('Exact prepared manifest/seed and fresh output required')
    manifest=json.loads((base/'prepared.json').read_text())
    if (not manifest['cpu_references_complete'] or manifest['gpu_admitted']
            or manifest['train_updates']!=0 or set(manifest['seeds'])!={'0','1'}):
        raise ValueError('Expected completed CPU-only initialization')
    root=base/('s'+str(seed));pin=manifest['seeds'][str(seed)]['config_sha256']
    t,meta=load_prepared(root,pin,device=device)
    if meta['seed']!=seed or meta['fixture_only']!=manifest['fixture_only']:
        raise ValueError('Prepared seed/fixture mismatch')
    if device=='cpu' and not meta['fixture_only']:
        raise ValueError('CPU driver is fixture-only')
    if device=='cuda' and (meta['fixture_only'] or fixture_steps is not None):
        raise ValueError('Actual CUDA qualification requires40 updates')
    steps=40 if device=='cuda' else fixture_steps
    if type(steps) is not int or not 2<=steps<=40 or steps%2:
        raise ValueError('Even bounded qualification length required')
    numerical=check_references(t,meta,root)
    # Capture initialization after pure references; none of the probe updates
    # ever overwrite it. Same baseline/RNG/data for whole and split runs.
    initial=copy.deepcopy(t.state());data=t.data;config=t.model.c
    del t;gc.collect()
    def fresh():
        n=DeviceTrainer(config,data,seed=seed,device=device)
        n.recipe=copy.deepcopy(initial['recipe'])
        n._restore_state(initial,meta['initial_sha256'],None)
        restore_rng(initial,device)
        return n
    out.mkdir(parents=True)
    (out/'numerical.json').write_text(json.dumps(numerical,indent=2)+'\n')
    gates={}
    for label in SEGMENTS:
        whole=run(fresh(),label,out/(label+'-whole.pt'),steps)
        part=run(fresh(),label,out/(label+'-part.pt'),steps//2)
        resumed=run(fresh(),label,out/(label+'-resumed.pt'),steps,resume=part)
        a=torch.load(whole['path'],map_location='cpu',weights_only=False)
        b=torch.load(resumed['path'],map_location='cpu',weights_only=False)
        exact(a,b)
        gates[label]=dict(exact_resume=True,updates_per_lineage=steps,
                          actual_updates_executed=steps*2,whole=whole,resumed=resumed)
        del a,b;gc.collect()
        print('ANCHOR_SEGMENT_COMPLETE',seed,label,flush=True)
    # Check the production-lineage warm-up kernel against the segment kernel.
    n=fresh();n.advance(steps)
    a=torch.load(gates['single']['whole']['path'],map_location='cpu',weights_only=False)
    state=n.state()
    for key in a['state']:exact(state[key],a['state'][key])
    exact([{k:v for k,v in row.items() if k!='step'} for row in n.trace],a['record']['trace'])
    path=out/'warmup-lineage.pt';lineage_sha=n.save(path)
    result=dict(schema=1,seed=seed,fixture_only=meta['fixture_only'],device=device,
                prepared_sha256=prepared_sha,config_sha256=pin,source=device_sources(),
                numerical_sha256=sha(out/'numerical.json'),segments=gates,
                lineage_matches_single_segment=True,warmup_lineage_sha256=lineage_sha,
                actual_updates_executed=steps*11,production_updates_executed=0,
                production_fit_admitted=False,
                limitation='Segment probes start at initialization; not trained phase-boundary states')
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True)
    p.add_argument('--sha',required=True);p.add_argument('--seed',type=int,choices=(0,1),required=True)
    p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4)
    print('ANCHOR_GPU_QUALIFICATION_COMPLETE',json.dumps(qualify_seed(a.prepared,a.sha,a.seed,a.out)),flush=True)


if __name__=='__main__':main()
