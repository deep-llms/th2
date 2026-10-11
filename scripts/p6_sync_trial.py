"""Bounded exact-state audit and complete-update timing for A/P6 synchronization."""
import argparse
import hashlib
import json
import os
import math
from pathlib import Path
import runpy
import statistics
import sys

from scripts.p6_compile_trial import read, write


def make_jobs(root):
    """Reuse original checkpoint recipe for audits, normal recipe for timings."""
    import copy
    from deep_kv.__main__ import jobs
    from run_experiments import load_jobs
    root=Path(root);project=Path.cwd()
    source=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed1042-20261009-a01/supervised/run/seed-1042/P6-iso')
    write(root/'sources.json',{'P6-iso':str(source)})
    saved=read(source/'train_config.json')
    recipe={**saved['model'],**saved['data'],**saved['pilot'],**saved['training']}
    items=[]
    for mode in ('reference','optimized'):
        folder='{run_dir}/audit/'+mode
        items.append(dict(name='stage-'+mode,argv=['{python}','-m','scripts.stage_deep_kv_resume',
            '--sources',str(root/'sources.json'),'--destination',folder,'--step','10000'],
            required_outputs=[dict(path='audit/'+mode+'/resume_inputs.json',json_equals=dict(status='ok'))]))
        # JSON preserves the exact source TrainingArguments, including lists.
        invocation={**recipe,'output_dir':str(root/'supervised/run/audit'/mode/'P6-iso'),
                    'stop_after':10002,'proxy_sync_optimization':mode=='optimized',
                    'allow_performance_change_on_resume':True,'save_total_limit':0}
        path=root/f'{mode}.json';write(path,invocation)
        items.append(dict(name='audit-'+mode,gpus=list(range(8)),timeout_seconds=2400,
            argv=['{python}','-m','accelerate.commands.launch','--config_file',str(project/'resources/accelerate_config.yaml'),
                  '--module','scripts.p6_sync_trial','train','--exact-audit','--',str(path)],
            required_outputs=[dict(path=f'audit/{mode}/P6-iso/sync-audit-rank{rank}.json',json_equals=dict(status='complete')) for rank in range(8)]))
    items.append(dict(name='exact-gate',argv=['{python}','-m','scripts.p6_sync_trial','gate','--root','{run_dir}/audit'],
        required_outputs=[dict(path='audit/verified.json',json_equals=dict(status='passed',exact=True))]))
    recipe=read(root/'recipe.json')
    recipe['proxy_module_seed']=1043
    names=('A-reference','P6-reference','P6-optimized','A-optimized',
           'A-optimized-repeat','P6-optimized-repeat','P6-reference-repeat','A-reference-repeat')
    for name in names:
        arm='A' if name.startswith('A-') else 'P6-iso'
        path=root/(name+'.json');write(path,{**recipe,'proxy_sync_optimization':'optimized' in name})
        job=copy.deepcopy(next(j for j in jobs(path,stop_after=100,arms=[arm],seeds=[1042])['jobs'] if 'gpus' in j))
        job['name']=name;job['timeout_seconds']=3600
        job['argv']=[v.replace('{run_dir}','{run_dir}/'+name) for v in job['argv']]
        for out in job['required_outputs']:out['path']=name+'/'+out['path']
        index=job['argv'].index(str(project/'train.py'))
        job['argv'][index:index+1]=['--module','scripts.p6_sync_trial','train','--']
        for rank in range(8):
            for prefix in ('sync-audit','step-profile'):
                job['required_outputs'].append(dict(path=f'{name}/seed-1042/{arm}/{prefix}-rank{rank}.json'))
        items.append(job)
    items.append(dict(name='summary',argv=['{python}','-m','scripts.p6_sync_trial','summary','--root','{run_dir}'],
        required_outputs=[dict(path='summary.json',json_equals=dict(status='passed'))]))
    write(root/'jobs.json',dict(jobs=items));load_jobs(root/'jobs.json')
    return items


def train(args, rest):
    import torch
    import random
    import numpy as np
    from transformers import TrainerCallback
    import deep_kv.proxy_training as training
    from scripts.check_optimized_resume import state_hash
    from scripts.proxy_speed_validation import deterministic_fa4
    evidence = dict(batches=[], gradients=[])
    class Audit(TrainerCallback):
        def __init__(self, trainer): self.trainer=trainer
        def on_pre_optimizer_step(self, arguments, state, control, **kwargs):
            if args.exact_audit:
                evidence['gradients'].append(state_hash({k:p.grad for k,p in self.trainer.model.named_parameters()}))
        def on_train_end(self, arguments, state, control, **kwargs):
            evidence.update(status='complete',global_step=state.global_step,rank=arguments.process_index)
            if args.exact_audit:
                evidence.update(model=state_hash(self.trainer.model.state_dict()),
                    optimizer=state_hash(self.trainer.optimizer.state_dict()),
                    scheduler=state_hash(self.trainer.lr_scheduler.state_dict()))
            write(Path(arguments.output_dir)/f'sync-audit-rank{arguments.process_index}.json',evidence)
    initialize,step=training.ProxyTrainer.__init__,training.ProxyTrainer.training_step
    restore=training.ProxyTrainer._load_optimizer_and_scheduler
    rng=training.ProxyTrainer._load_rng_state
    def init(self,*a,**kw):
        initialize(self,*a,**kw);self.add_callback(Audit(self))
    def load(self,checkpoint):
        restore(self,checkpoint)
        if args.exact_audit:
            from safetensors.torch import load_file
            expected_model=load_file(str(Path(checkpoint)/'model.safetensors'))
            evidence['restored_model']=state_hash(self.model.state_dict())
            assert evidence['restored_model']==state_hash(expected_model),'Model restore mismatch'
            for name,actual in [('optimizer',self.optimizer.state_dict()),('scheduler',self.lr_scheduler.state_dict())]:
                expected=torch.load(Path(checkpoint)/f'{name}.pt',map_location='cpu',weights_only=True)
                assert state_hash(expected)==state_hash(actual),name+' restore mismatch'
                evidence['restored_'+name]=state_hash(actual)
    def load_rng(self,checkpoint):
        rng(self,checkpoint)
        if args.exact_audit:
            suffix=f'_{self.args.process_index}' if self.args.world_size>1 else ''
            expected=torch.load(Path(checkpoint)/f'rng_state{suffix}.pth',weights_only=False)
            actual=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state())
            if 'cuda' in expected:actual['cuda']=torch.cuda.get_rng_state_all() if self.args.world_size>1 else torch.cuda.get_rng_state()
            assert state_hash(expected)==state_hash(actual),'RNG restore mismatch'
            evidence['restored_rng']=state_hash(actual)
    def training_step(self,model,inputs,*a,**kw):
        if args.exact_audit or self.state.global_step in (0,99):
            evidence['batches'].append(state_hash(inputs))
        return step(self,model,inputs,*a,**kw)
    training.ProxyTrainer.__init__=init
    training.ProxyTrainer.training_step=training_step
    training.ProxyTrainer._load_optimizer_and_scheduler=load
    training.ProxyTrainer._load_rng_state=load_rng
    os.environ['P6_PROFILE_UPDATE']='0'
    sys.argv=['train.py',*(rest[1:] if rest and rest[0]=='--' else rest)]
    with deterministic_fa4(args.exact_audit):
        runpy.run_module('scripts.profile_proxy_training' if not args.exact_audit else 'train',run_name='__main__')


def gate(root):
    root=Path(root)
    for rank in range(8):
        a,b=[read(root/m/'P6-iso'/f'sync-audit-rank{rank}.json') for m in ('reference','optimized')]
        for key in ('restored_model','restored_optimizer','restored_scheduler','restored_rng','batches','gradients','model','optimizer','scheduler'):
            if key not in a or not a[key] or a[key]!=b[key]:raise ValueError(f'Exact GPU gate failed: {key}, rank {rank}')
        if a['global_step']!=10002 or b['global_step']!=10002 or len(a['batches'])!=8 or len(a['gradients'])!=2:
            raise ValueError('Incomplete two-update resume audit')
    write(root/'verified.json',dict(status='passed',exact=True,world_size=8,updates=2,
          note='Identical restored optimizer/scheduler/RNG, batches, clipped gradients and final model/optimizer/scheduler. Deterministic FA4.'))


def summary(root):
    root=Path(root);runs=[]
    for name in ('A-reference','P6-reference','P6-optimized','A-optimized','A-optimized-repeat','P6-optimized-repeat','P6-reference-repeat','A-reference-repeat'):
        arm='A' if name.startswith('A-') else 'P6-iso'
        folder=root/name/'seed-1042'/arm
        config,result=read(folder/'train_config.json'),read(folder/'result.json')
        reports=[read(folder/f'step-profile-rank{r}.json') for r in range(8)]
        audits=[read(folder/f'sync-audit-rank{r}.json') for r in range(8)]
        if result['global_step']!=100 or any(len(a['batches'])!=8 for a in audits):raise ValueError('Incomplete timing/audit')
        hashes=[a['batches'] for a in audits]
        if runs and (hashes!=runs[0]['batches'] or config['train_fingerprint']!=runs[0]['train_fingerprint']):raise ValueError('Data mismatch')
        # Max across ranks, includes ALL callback/logging/data-delivery work.
        times=[max(next(s['interval_seconds'] for s in r['steps'] if s['step']==i) for r in reports) for i in range(21,91)]
        if not all(math.isfinite(t) and t>0 for t in times):raise ValueError('Invalid timing')
        runs.append(dict(name=name,mean_seconds=statistics.mean(times),median_seconds=statistics.median(times),
            times=times,peak_gib=max(r['peak_gib'] for r in reports),batches=hashes,
            train_fingerprint=config['train_fingerprint'],evaluation=result['evaluation']))
    means={key:statistics.mean(r['mean_seconds'] for r in runs if r['name'] in (key,key+'-repeat'))
           for key in ('A-reference','P6-reference','A-optimized','P6-optimized')}
    for arm in ('A-','P6-'):
        values=[r['evaluation']['eval_lm_loss'] for r in runs if r['name'].startswith(arm)]
        if not all(math.isfinite(v) for v in values) or max(values)-min(values)>.002:
            raise ValueError('Short production-FA4 loss trajectories differ beyond the declared 0.002 NLL gate')
    write(root/'summary.json',dict(status='passed',runs=runs,mean_seconds=means,
        optimized_overhead=means['P6-optimized']/means['A-optimized']-1,
        reference_overhead=means['P6-reference']/means['A-reference']-1,
        note='Eight real 100-step Trainer fits, reverse-order repeats. Complete start-to-start intervals 21–90; max across eight ranks. No compilation.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--exact-audit',action='store_true')
    for name in ('gate','summary'):sub.add_parser(name).add_argument('--root',required=True)
    args,rest=p.parse_known_args()
    if args.command=='train':train(args,rest)
    elif rest:p.error('Unexpected arguments')
    else:globals()[args.command](args.root)
