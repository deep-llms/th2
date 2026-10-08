"""Disposable proxy optimization checks; never imported by ordinary training."""
import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'resources/proxy_reference_4f8208b'
NEW_ARMS = ('P6-iso-sparse', 'P6-iso-short', 'P6-iso-weighted', 'P6-iso-layernorm',
            'P7-simple-sparse', 'P7-simple-short')
BENCH_ARMS = ('A', 'P6-iso', 'P7-simple', 'P7')
# Eight FP32 epsilons; far below BF16 rounding. No output tolerance.
ROUNDING_LIMIT = 2**-20


def read(path):return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


@contextmanager
def implementation(name):
    """Select archived modules only inside this validation process."""
    if name == 'optimized':
        yield
        return
    if name != 'previous':raise ValueError(name)
    manifest=read(REFERENCE/'manifest.json')
    modules=('proxy_memory','proxy','proxy_training')
    saved={}
    parent=importlib.import_module('deep_kv')
    try:
        for short in modules:
            canonical='deep_kv.'+short
            saved[short]=importlib.import_module(canonical)
            path=REFERENCE/(short+'.py')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][path.name]
            spec=importlib.util.spec_from_file_location(canonical,path)
            module=importlib.util.module_from_spec(spec)
            sys.modules[canonical]=module;setattr(parent,short,module)
            spec.loader.exec_module(module)
        yield
    finally:
        for short,module in saved.items():
            sys.modules['deep_kv.'+short]=module;setattr(parent,short,module)


def exact_tensors(a,b):
    import torch
    if a.keys()!=b.keys():raise AssertionError('Tensor keys differ')
    failures=[]
    for key,x in a.items():
        y=b[key]
        if x.shape!=y.shape or x.dtype!=y.dtype:
            failures.append(dict(key=key,reason='shape or dtype differs'));continue
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            failures.append(dict(key=key,reason='nonfinite'));continue
        if not torch.equal(x,y):
            delta=(x.double()-y.double())
            failures.append(dict(key=key,max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm()/x.double().norm().clamp_min(1e-30))))
    return failures


def rounding_violations(a, differences):
    """Bound both per-tensor relative L2 and peak-scaled absolute error."""
    return [d for d in differences if 'reason' in d or
            d['relative_l2'] > ROUNDING_LIMIT or
            d['max_abs'] > ROUNDING_LIMIT*float(a[d['key']].abs().max())]


@contextmanager
def deterministic_fa4(enabled):
    """Validation-process override only; production defaults are untouched."""
    if not enabled:
        yield
        return
    from functools import partial
    import inspect
    from deep_kv import fa4
    original=fa4.load_kernel
    def load():
        kernel,metadata=original()
        assert 'deterministic' in inspect.signature(kernel).parameters
        return partial(kernel,deterministic=True),{**metadata,'deterministic':True,'validation_only':True}
    fa4.load_kernel=load
    try:yield
    finally:fa4.load_kernel=original


def capture(arm, backend, checkpoint, recipe, rows):
    import torch
    from transformers import AutoConfig
    from deep_kv.proxy import ProxyModel
    from deep_kv.proxy_training import ProxyTrainer
    from deep_kv.packing import isolated_data_collator
    from scripts.check_trained_attention import fingerprint
    cfg=AutoConfig.from_pretrained(recipe['config_name'],local_files_only=True)
    cfg._attn_implementation='sdpa';cfg.use_cache=False
    deterministic=backend=='fa4_deterministic'
    m=ProxyModel.from_scratch(cfg,arm,attention_backend='fa4' if deterministic else backend,seed=42,
        checkpoint_layers=checkpoint,checkpoint_aux=checkpoint,checkpoint_lm=checkpoint,lm_chunk=128).cuda().train()
    if deterministic:
        from functools import partial
        import inspect
        assert 'deterministic' in inspect.signature(m.fa4_kernel).parameters
        m.fa4_kernel=partial(m.fa4_kernel,deterministic=True)
    ctx=ProxyTrainer.context({k:v.cuda() for k,v in isolated_data_collator(rows[:2]).items()})
    assert ctx.input_ids.shape==(2,2048) and len(set(rows[0]['segments']))>1
    initial=fingerprint(m)
    outputs={}
    with torch.no_grad():
        for phase in ('mean','variance'):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=m(ctx,compute_auxiliary_losses=False,auxiliary_grad=False,
                      collect_target_statistics=True,statistics_mode=phase)
            outputs.update({phase+'/'+k:v.detach().cpu().clone() for k,v in out.items()})
            m.update_statistics(out['center_sums'],out['center_squares'],out['center_counts'],initialize=phase)
    def hidden(module,inputs,value):
        outputs['hidden']=value.detach().float().cpu()
    handle=m.backbone.model.norm.register_forward_hook(hidden)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=m(ctx,collect_target_statistics=True,relational_step=250)
        loss=out['lm_sum']/out['lm_count']+.1*out['aux_sum']/out['aux_count'].clamp_min(1)
        if m.relational_proxy:loss=loss+.05*out['rel_sum']/out['rel_count'].clamp_min(1)
    loss.backward();handle.remove()
    outputs.update({k:v.detach().cpu().clone() for k,v in out.items()})
    gradients={k:p.grad.detach().float().cpu().clone() for k,p in m.named_parameters() if p.grad is not None}
    assert len(gradients)==len(list(m.named_parameters()))
    m.update_statistics(out['center_sums'],out['center_squares'],out['center_counts'])
    outputs.update({'buffer/'+k:v.detach().cpu().clone() for k,v in m.named_buffers()})
    result=dict(initial=initial,outputs=outputs,gradients=gradients,loss=float(loss.detach()),
                peak_gib=torch.cuda.max_memory_allocated()/2**30)
    del m,out,loss,ctx
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    return result


def gate(args):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    recipe,rows=read(args.recipe),read(args.rows)
    cases=[]
    for backend in ('sdpa_math','fa4_deterministic'):
        for checkpoint in (False,True):
            baseline=None
            for name in ('previous','optimized'):
                print('CAPTURE',args.arm,backend,name,checkpoint,flush=True)
                context=sdpa_kernel([SDPBackend.MATH]) if backend=='sdpa_math' else nullcontext()
                with context,implementation(name):
                    value=capture(args.arm,'sdpa' if backend=='sdpa_math' else backend,checkpoint,recipe,rows)
                if baseline is None:baseline=value
                assert baseline['initial']==value['initial'],'Initial weights/buffers differ'
                errors={k:exact_tensors(baseline[k],value[k]) for k in ('outputs','gradients')}
                differences=errors
                if backend=='fa4_deterministic':
                    errors={**errors,'gradients':rounding_violations(baseline['gradients'],errors['gradients'])}
                case=dict(backend=backend,implementation=name,checkpoint=checkpoint,errors=errors,differences=differences,
                          loss=value['loss'],peak_gib=value['peak_gib'])
                cases.append(case)
                if any(errors.values()):
                    write(args.output,dict(status='failed',arm=args.arm,cases=cases,rounding_limit=ROUNDING_LIMIT))
                    raise AssertionError('CUDA comparison failed; evidence saved')
                if value is not baseline:del value
            del baseline
    write(args.output,dict(status='passed',arm=args.arm,cases=cases,
                          criterion='Exact outputs and math-SDPA gradients; deterministic FA4 gradients bounded by eight FP32 epsilons',
                          rounding_limit=ROUNDING_LIMIT,
                          reference_commit=read(REFERENCE/'manifest.json')['commit']))


def training(args, rest):
    if rest and rest[0]=='--':rest=rest[1:]
    sys.argv=['train.py',*rest]
    assert not (args.profile and args.deterministic_fa4),'Timing must use production FA4'
    with deterministic_fa4(args.deterministic_fa4),implementation(args.implementation):
        records=[];location=None
        if args.audit_update25:
            from deep_kv.proxy_training import ProxyTrainer
            original=ProxyTrainer.training_step
            def audited(self,model,inputs,*pos,**kw):
                nonlocal location
                if self.state.global_step==24:
                    row={}
                    for key,value in sorted(inputs.items()):
                        tensor=value.detach().cpu().contiguous()
                        row[key]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),
                                      sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
                    records.append(row)
                    location=Path(self.args.output_dir)/f'update25-rank{self.args.process_index}.json'
                return original(self,model,inputs,*pos,**kw)
            ProxyTrainer.training_step=audited
        runpy.run_module('scripts.profile_proxy_training' if args.profile else 'train',run_name='__main__')
        if args.audit_update25:
            assert len(records)==4 and location is not None
            write(location,dict(step=25,microbatches=records))


def gate_all(args):
    """Independent numerical checks share the eight GPUs; training never does."""
    from concurrent.futures import ThreadPoolExecutor
    import os
    import subprocess
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(visible)==8 and len(set(visible))==8
    arms=(*NEW_ARMS,'P6-iso','P7-simple','P7')
    root=Path(args.output).parent;root.mkdir(parents=True,exist_ok=True)
    def slot(index):
        results=[]
        for arm in arms[index::8]:
            report=root/(arm+'.json')
            with (root/(arm+'.log')).open('x') as log:
                command=[sys.executable,'-u','-m','scripts.proxy_speed_validation','gate',
                         '--arm',arm,'--recipe',str(args.recipe),'--rows',str(args.rows),'--output',str(report)]
                completed=subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':visible[index]},
                                         stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            passed=completed.returncode==0 and report.is_file() and read(report)['status']=='passed'
            results.append(dict(arm=arm,gpu=visible[index],returncode=completed.returncode,passed=passed))
            print('NUMERICAL_GATE',arm,'passed' if passed else 'failed',flush=True)
        return results
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=[item for group in pool.map(slot,range(8)) for item in group]
    passed=all(item['passed'] for item in results)
    write(args.output,dict(status='passed' if passed else 'failed',cases=results))
    assert passed,'Numerical gate failed; see individual reports/logs'


def diagnose(args):
    """Measure unchanged-code repeatability before interpreting a gate failure."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    recipe,rows=read(args.recipe),read(args.rows);cases=[]
    for backend in getattr(args,'backends',('sdpa','sdpa_math','fa4')):
        baseline=None
        for label,name in (('reference','previous'),('repeat_reference','previous'),('optimized','optimized')):
            print('REPEATABILITY',backend,label,flush=True)
            context=sdpa_kernel([SDPBackend.MATH]) if backend=='sdpa_math' else nullcontext()
            with context,implementation(name):
                value=capture(args.arm,'sdpa' if backend=='sdpa_math' else backend,False,recipe,rows)
            if baseline is None:baseline=value
            assert baseline['initial']==value['initial']
            comparisons={k:exact_tensors(baseline[k],value[k]) for k in ('outputs','gradients')}
            cases.append(dict(backend=backend,label=label,differences=comparisons,loss=value['loss'],peak_gib=value['peak_gib']))
            if value is not baseline:del value
        del baseline
    write(args.output,dict(status='measured',arm=args.arm,cases=cases,
                          note='Repeatability diagnostic, not a relaxed acceptance gate.'))


def copy_resume(args):
    source,destination=Path(args.source),Path(args.destination)
    assert not destination.exists()
    checkpoint=source/'checkpoint-24'
    assert read(checkpoint/'trainer_state.json')['global_step']==24
    destination.mkdir(parents=True)
    shutil.copy2(source/'train_config.json',destination/'train_config.json')
    shutil.copytree(checkpoint,destination/checkpoint.name)
    names=('model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json',*[f'rng_state_{i}.pth' for i in range(8)])
    hashes={}
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        return h.hexdigest()
    for name in names:
        hashes[name]=digest(checkpoint/name)
        assert hashes[name]==digest(destination/checkpoint.name/name)
    write(args.output,dict(status='passed',source=str(checkpoint),destination=str(destination),sha256=hashes))


def check_resume(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    a,b=Path(args.source)/'checkpoint-25',Path(args.destination)/'checkpoint-25'
    for p in (a,b):assert read(p/'trainer_state.json')['global_step']==25
    for p in (a,b):assert read(p.parent/'result.json')['attention_runtime']['deterministic'] is True
    for rank in range(8):
        name=f'update25-rank{rank}.json'
        assert read(a.parent/name)==read(b.parent/name),'Resumed data differs: '+name
    original=load_file(a/'model.safetensors');resumed=load_file(b/'model.safetensors')
    differences=exact_tensors(original,resumed)
    errors=rounding_violations(original,differences)
    # Normalization targets before the update must be exactly identical.
    for key in ('mu','sigma2','mu_initialized'):
        assert torch.equal(original[key],resumed[key]),key
    def compare(x,y,path):
        if isinstance(x,torch.Tensor):
            d=exact_tensors({path:x},{path:y});differences.extend(d)
            errors.extend(rounding_violations({path:x},d) if x.is_floating_point() and path.startswith('optimizer.pt/state/') else d)
        elif isinstance(x,np.ndarray):
            assert np.array_equal(x,y),path
        elif isinstance(x,dict):
            assert x.keys()==y.keys(),path
            for key in x:compare(x[key],y[key],path+'/'+str(key))
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y),path
            for i,(v,w) in enumerate(zip(x,y)):compare(v,w,path+'/'+str(i))
        else:assert x==y,path
    for name in ('optimizer.pt','scheduler.pt',*[f'rng_state_{i}.pth' for i in range(8)]):
        # These are checkpoints just produced by this disposable trusted queue.
        compare(torch.load(a/name,map_location='cpu',weights_only=False),
                torch.load(b/name,map_location='cpu',weights_only=False),name)
    write(args.output,dict(status='passed' if not errors else 'failed',errors=errors,differences=differences,step=25,
                          checkpoint_step=24,rounding_limit=ROUNDING_LIMIT,
                          criterion='Exact data, normalization, scheduler and RNG; model/optimizer within eight FP32 epsilons'))
    assert not errors,'Resume differs; evidence saved'


def summary(args):
    root=Path(args.run_dir);values={}
    for arm in BENCH_ARMS:
        values[arm]={}
        for name in ('previous','optimized'):
            folder=root/'benchmark'/name/'seed-42'/arm
            result=read(folder/'result.json');timing=read(folder/'step-profile.json')
            assert not result['attention_runtime'].get('deterministic',False)
            times=[r['seconds'] for r in timing['steps'] if 10<=r['step']<=25]
            assert len(times)==16 and all(x>0 for x in times)
            profile=read(folder/'component-profile.json');assert profile['kernel_milliseconds']
            values[arm][name]=dict(median_seconds=statistics.median(times),step_seconds=times,
                peak_allocated_gib=result['training_cost']['peak_cuda_allocated_bytes']/2**30,
                peak_reserved_gib=result['training_cost']['peak_cuda_reserved_bytes']/2**30,
                component_kernel_ms=profile['kernel_milliseconds'])
        values[arm]['optimized_time_ratio']=values[arm]['optimized']['median_seconds']/values[arm]['previous']['median_seconds']
    write(args.output,dict(status='passed',benchmark=values,reference_commit=read(REFERENCE/'manifest.json')['commit'],
        note='One paired 25-step run per arm/implementation; steps 10–25. Kernel sums are not wall time.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
    for mode in ('gate','diagnose'):
        g=s.add_parser(mode)
        for name in ('arm','recipe','rows','output'):g.add_argument('--'+name,required=True)
        if mode=='diagnose':g.add_argument('--backends',nargs='+',choices=('sdpa','sdpa_math','fa4','fa4_deterministic'),default=('sdpa','sdpa_math','fa4'))
    g=s.add_parser('gate-all')
    for name in ('recipe','rows','output'):g.add_argument('--'+name,required=True)
    t=s.add_parser('train');t.add_argument('--implementation',choices=('previous','optimized'),required=True)
    t.add_argument('--profile',action='store_true')
    t.add_argument('--audit-update25',action='store_true')
    t.add_argument('--deterministic-fa4',action='store_true')
    for name in ('copy-resume','check-resume'):
        q=s.add_parser(name)
        for key in ('source','destination','output'):q.add_argument('--'+key,required=True)
    q=s.add_parser('summary');q.add_argument('--run-dir',required=True);q.add_argument('--output',required=True)
    args,rest=p.parse_known_args()
    if args.mode=='train':training(args,rest)
    else:
        if rest:p.error('Unrecognized arguments: '+repr(rest))
        {'gate':gate,'gate-all':gate_all,'diagnose':diagnose,'copy-resume':copy_resume,'check-resume':check_resume,'summary':summary}[args.mode](args)


if __name__=='__main__':main()
