"""Frozen-backbone global-context budget screen; matched native-Trainer router fit."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from efficient_llm.context_attention import ContextPolicy
from efficient_llm.context_check import check
from efficient_llm.diagnostic import load_model, load_probe, evaluate, paired_summary, write
from efficient_llm.diagnostic_data import sha256
from efficient_llm.group_selection import subset_masks
from efficient_llm.router_pilot import fit_routers


def static_policies(cost_sum, totals, groups, keep):
    if not torch.isfinite(totals).all() or (totals[:,1]<=0).any():raise ValueError('Invalid layer calibration')
    fixed=subset_masks(groups,keep,device=cost_sum.device)[cost_sum[:,0].argmin(-1)]
    layers=len(totals); count=layers*keep//groups
    if layers*keep%groups:raise ValueError('Whole-layer control requires integral budget')
    scores=totals[:,0]/totals[:,1]
    chosen=scores.argsort(descending=True,stable=True)[:count]
    whole=torch.zeros_like(fixed);whole[chosen]=True
    periodic=torch.zeros_like(fixed)
    # Uniformly spaced global layers; exact matched total group budget.
    periodic[(torch.arange(count,device=cost_sum.device)*layers//count).long()]=True
    return {'fixed_groups':fixed,'fixed_layers':whole,'periodic_layers':periodic},scores


def policy(model,name,config,routers=None,reference=False):
    kwargs={k:config[k] for k in ('keep','window','sinks')}
    if name in config['static_masks']:
        return ContextPolicy(model,mode='reference_static' if reference else 'static',
            static_masks=torch.tensor(config['static_masks'][name],device=next(model.parameters()).device,dtype=torch.bool),**kwargs)
    if name=='local':return ContextPolicy(model,mode='local',**kwargs)
    if name=='oracle':return ContextPolicy(model,mode='oracle',**kwargs)
    if name in ('joint_router','independent_router'):
        return ContextPolicy(model,mode='router',routers=routers,bank=int(name=='independent_router'),**kwargs)
    raise ValueError('Unknown context method: '+name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','probe','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--long-probe',type=Path)
    for name,default in [('keep',4),('window',128),('sinks',4),('stride',16),('steps',512),('feature-batch',256)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0))) if a.device=='cuda' else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    try:
        if min(a.keep,a.window,a.stride,a.steps,a.feature_batch)<1 or a.sinks<0:raise ValueError('Invalid recipe')
        if rank==0:
            a.output.mkdir(parents=True,exist_ok=False);(a.output/'features').mkdir()
        if world>1:dist.barrier()
        manifest,tokens=load_probe(a.probe)
        if rank==0:
            for name,expected in manifest['model_files'].items():
                if sha256(a.model/name)!=expected:raise ValueError('Model identity changed')
        if world>1:dist.barrier()
        torch.manual_seed(42);torch.set_float32_matmul_precision('highest')
        dtype=torch.bfloat16 if device.type=='cuda' else torch.float32
        model=load_model(a.model,device,dtype)
        teacher=ContextPolicy(model,mode='teacher',keep=a.keep,window=a.window,sinks=a.sinks,stride=a.stride)
        routers,cost_sum=fit_routers(model,tokens,output=a.output,probe=a.probe,keep=a.keep,stride=a.stride,
            steps=a.steps,feature_batch=a.feature_batch,rank=rank,world=world,teacher=teacher)
        if world>1:dist.all_reduce(teacher.layer_totals)
        masks,scores=static_policies(cost_sum,teacher.layer_totals,model.config.num_key_value_heads,a.keep)
        config=dict(protocol='context_budget_v1',keep=a.keep,window=a.window,sinks=a.sinks,
            static_masks={k:v.cpu().tolist() for k,v in masks.items()},layer_local_error_over_energy=scores.cpu().tolist(),
            model_files=manifest['model_files'],routers_sha256=sha256(a.output/'routers.pt'),
            probe_manifest_sha256=sha256(a.probe/'manifest.json'))
        if rank==0:write(a.output/'context-config.json',config)
        errors=check(model.float(),tokens['calibration'][:1,:min(32,manifest['length'])].to(device),routers)
        model.to(dtype=dtype)
        scopes={'development_2k':(a.probe,manifest,tokens)}
        if a.long_probe:
            long_manifest,long_tokens=load_probe(a.long_probe)
            if long_manifest['model_files']!=manifest['model_files']:raise ValueError('Long probe model differs')
            # Prefix scopes can overlap each other; neither may overlap fit documents.
            train_hashes={r['text_sha256'] for r in manifest['documents']['calibration']}
            eval_hashes={r['text_sha256'] for r in long_manifest['documents']['evaluation']}
            if train_hashes & eval_hashes:raise ValueError('Long probe overlaps router training')
            scopes['long_8k']=(a.long_probe,long_manifest,long_tokens)
        report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),world_size=world,
            config=config,fp32_checks=errors,precision=str(dtype),scopes={},
            limitations=['Frozen backbone; natural-English long-document prefixes, not full validation or retrieval benchmark.',
                '2k development probe reused from previous router pilot. Long probe can overlap its documents.',
                'Learned/oracle evaluation computes both branches; only static policies have execution-speed measurements.',
                'Full native KV retained: no KV-capacity reduction. Incremental comparison with existing local/global research.'])
        for scope,(path,meta,data) in scopes.items():
            methods={};report['scopes'][scope]=dict(length=meta['length'],probe_manifest_sha256=sha256(path/'manifest.json'),methods=methods)
            for name in ('dense','local','fixed_groups','fixed_layers','periodic_layers','oracle','joint_router','independent_router'):
                if world>1:dist.barrier()
                start=time.monotonic()
                metrics,_=evaluate(model,data['evaluation'],keep=a.keep,policy='dense',rank=rank,world=world,batch_size=1,
                    selection=None if name=='dense' else policy(model,name,config,routers))
                metrics['elapsed_seconds']=time.monotonic()-start;methods[name]=metrics
                if name!='dense':metrics['vs_dense']=paired_summary(metrics['per_document_nll'],methods['dense']['per_document_nll'])
                if rank==0:
                    print('CONTEXT_RESULT',json.dumps(dict(scope=scope,method=name,**{k:v for k,v in metrics.items() if k!='per_document_nll'})),flush=True)
                    write(a.output/'evaluation.partial.json',report)
            report['scopes'][scope]['joint_comparisons']={name:paired_summary(methods['joint_router']['per_document_nll'],methods[name]['per_document_nll'])
                for name in ('independent_router','fixed_groups','fixed_layers','oracle')}
        report.update(status='passed',finished_utc=datetime.now(timezone.utc).isoformat())
        if rank==0:write(a.output/'evaluation.json',report);print('CONTEXT_STUDY_COMPLETE',flush=True)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
