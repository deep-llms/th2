"""Bounded actual-input CUDA qualification; never launches full continuation.

The owned runner must supply one freshly reclaimed GPU. Fullstates and moments
remain on-node. A receipt proves numerical checks, not a scientific mechanism.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from .adaptation_train import train, load_parent, source_hashes
from .adaptation_reservation import validate_binding
from .adaptation_analysis import CANONICAL, SWAPPED, native_trace
from .adaptation_components import component_crossovers, score_crossovers
from .adaptation_runtime import measurement_runtime, measurement_policy, training_policy
from .adaptation_geometry import aligned_observations, GeometryMoments
from .adaptation_eval import evaluate, replay
from .data import Rows, sha
from .horizon_train import write_json


def exact(a,b, path="root"):
    """Semantic equality includes RNG/counters/moments, not serialization bytes."""
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a.cpu(),b.cpu()):
            raise ValueError('Fullstate tensor equality failed at '+path)
        if a.is_floating_point() and not torch.isfinite(a).all():raise ValueError('Nonfinite fullstate')
    elif isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray) or a.dtype!=b.dtype or not np.array_equal(a,b):raise ValueError('RNG array mismatch')
    elif isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():raise ValueError('Fullstate keys differ')
        for k in a:exact(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):raise ValueError('Fullstate sequence mismatch')
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
    elif a!=b:raise ValueError('Fullstate scalar mismatch')


def state(directory):
    directory=Path(directory);cp=json.loads((directory/'checkpoint.json').read_text())
    if sha(directory/'fullstate.pt')!=cp['sha256']:raise ValueError('Checkpoint byte hash mismatch')
    return torch.load(directory/'fullstate.pt',map_location='cpu',weights_only=False)


def qualify(config, prepared_path, prepared_sha256, out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh GPU qualification output required')
    if sha(prepared_path)!=prepared_sha256:raise ValueError('Prepared manifest mismatch')
    prepared=json.loads(Path(prepared_path).read_text());directory=Path(prepared_path).parent
    name=f"{config['branch']}-s{config['row_reservation']['seed']}.json"
    if sha(directory/name)!=prepared['files'][name] or json.loads((directory/name).read_text())!=config:
        raise ValueError('Exact prepared branch config required')
    validate_binding(config)
    if torch.cuda.device_count()!=1:raise ValueError('Exactly one visible qualification GPU required')
    torch.set_num_threads(config['threads'])
    out.mkdir(parents=True)
    model,opt,parent,names=load_parent(config['parent'],config['parent_hashes'])
    model.eval().requires_grad_(False)
    ids=np.load(config['calibration_indices'],allow_pickle=False)[:1]
    x=Rows(config['data'],model.c.length).batch(ids,'cpu')
    with torch.no_grad(),measurement_runtime('cpu'):
        reference={label:(model.hidden(x,path).clone(),model.nll(x,path).clone())
                   for label,path in [('canonical',CANONICAL),('swap',SWAPPED)]}
    model.to('cuda');x=x.to('cuda');parity={};effects={}
    with torch.no_grad(),measurement_runtime('cuda'):
        policy=measurement_policy()
        for label,path in [('canonical',CANONICAL),('swap',SWAPPED)]:
            h,trace=native_trace(model,x,path);nll=model.nll(x,path)
            torch.testing.assert_close(h.cpu(),reference[label][0],atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(nll.cpu(),reference[label][1],atol=1e-4,rtol=1e-4)
            parity[label]=dict(hidden_max_abs=float((h.cpu()-reference[label][0]).abs().max()),
                               nll_max_abs=float((nll.cpu()-reference[label][1]).abs().max()))
        for kind in ('attention','ffn'):
            for direction,donor,recipient in [('forward',CANONICAL,SWAPPED),('reverse',SWAPPED,CANONICAL)]:
                r=component_crossovers(model,x,2,kind,donor_path=donor,recipient_path=recipient)
                scored=score_crossovers(model,x,r);key=kind+'/'+direction
                parity[key]=r['parity_max_abs']
                effects[key]={k:v.tolist() for k,v in scored['effects_nll_sum'].items()}
                del r,scored
        # Exercise the same30-cell GPU scorer/weight crossovers and portable replay.
        # A and B intentionally equal the parent here: this is only the step0 gate.
        evaluate({'parent':model,'A':model,'B':model},x,ids,out/'zero-step-panel')
        if replay(out/'zero-step-panel')['cases']!=30:raise ValueError('GPU score pipeline gate')
        # Test actual GPU tracing -> bounded CPU moments -> independent raw replay.
        observations=aligned_observations(model,model,x,(31,127,255,511))
        geometry={}
        for key in ('slot/2/input','slot/2/output','block/2/input','block/2/output'):
            moment=GeometryMoments(model.c.d)
            moment.update(observations[key]['x'].cpu(),observations[key]['y'].cpu())
            arrays=moment.arrays();report=moment.report()
            if GeometryMoments.from_arrays(arrays).report()!=report:raise ValueError('Geometry replay mismatch')
            geometry[key]=report
            # Large sufficient statistics stay beside node result, excluded from small export.
            np.savez_compressed(out/(key.replace('/','_')+'_moments.npz'),**arrays)
    write_json(out/'numerical.json',dict(policy=policy,cpu_cuda_parity=parity,geometry=geometry,
        calibration_rows=ids.tolist(),component_effects_nll_sum=effects,
        scope='One calibration sequence; numerical qualification, not generalization/mechanism verdict'))
    del model,opt,reference,x,trace,observations,h,nll;gc.collect();torch.cuda.empty_cache()
    # Actual parent state supplies step0 identity. No optimizer reset or row reselection.
    for sub,stop,resume in [('whole',40,False),('split',20,False),('split',40,True)]:
        m,o=train(config,out/sub,stop_after=stop,resume=resume,device='cuda',gpu_qualification=True)
        del m,o;gc.collect();torch.cuda.empty_cache()
    zero=state(out/'whole/snapshots/000000')
    for key in ('model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng'):exact(zero[key],parent[key])
    del zero,parent
    traces={sub:[json.loads(l) for l in (out/sub/'trace.jsonl').read_text().splitlines()] for sub in ('whole','split')}
    mismatch=next((i+1 for i,(a,b) in enumerate(zip(traces['whole'],traces['split'])) if a!=b),None)
    write_json(out/'resume-diagnostic.json',dict(first_trace_mismatch=mismatch,
        trace_lengths={k:len(v) for k,v in traces.items()},
        training_policy=training_policy('cuda')))
    whole=state(out/'whole');split=state(out/'split');exact(whole,split)
    if whole['step']!=40:raise ValueError('Qualification not at40')
    for p in (out/'whole/trace.jsonl',out/'split/trace.jsonl'):
        trace=[json.loads(l) for l in p.read_text().splitlines()]
        if [r['step'] for r in trace]!=list(range(1,41)):raise ValueError('Missing first40 trace')
    if (out/'whole/trace.jsonl').read_bytes()!=(out/'split/trace.jsonl').read_bytes():raise ValueError('First40 traces differ')
    result=dict(step=40,exact_resume=True,finite=True,parent_step0_exact=True,
        parent_hashes=config['parent_hashes'],source=source_hashes(),torch=torch.__version__,
        training_policy=training_policy('cuda'),branch=config['branch'],seed=config['row_reservation']['seed'],precision='bf16',tf32=False,
        reservation_sha256=config['row_reservation']['manifest_sha256'],native_component_parity_pass=True,
        numerical_sha256=sha(out/'numerical.json'),prepared_sha256=prepared_sha256,
        whole_fullstate_sha256=sha(out/'whole/fullstate.pt'),split_fullstate_sha256=sha(out/'split/fullstate.pt'),
        qualification_updates_executed=80,production_updates_executed=0,weights_exportable=False)
    write_json(out/'complete.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--prepared',required=True)
    p.add_argument('--prepared-sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(qualify(json.loads(Path(a.config).read_text()),a.prepared,a.prepared_sha256,a.out)),flush=True)


if __name__=='__main__':main()
