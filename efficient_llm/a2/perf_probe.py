"""Discarded full-model 8-B200 performance probe; never exports a healed arm."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
import numpy as np
from transformers import TrainerCallback,default_data_collator,set_seed
from .data import PackedRows,write_json
from .model import KDModel,load_parent,prune
from .performance import SUITES,PerformanceKDModel,PerformanceTrainer
from .protocol import FIXED_SETS,layers_for,lr_for_update
from .train import HealTrainer,make_arguments,audit_order
from .measure import score


def compare_tensors(actual,reference):
    if set(actual)!=set(reference):raise ValueError('Parameter keys differ')
    sq=ref_sq=dot=actual_sq=0.;maximum=0.
    for key,g in actual.items():
        r=reference[key]
        if g.shape!=r.shape or not torch.isfinite(g).all():raise ValueError('Gradient invalid')
        # CPU FP64 scalar reductions, no full-model concatenation.
        diff=g.double()-r.double();sq+=diff.square().sum().item()
        ref_sq+=r.double().square().sum().item();actual_sq+=g.double().square().sum().item()
        dot+=(g.double()*r.double()).sum().item();maximum=max(maximum,diff.abs().max().item())
    return dict(relative_l2=(sq/max(ref_sq,1e-30))**.5,
        cosine=dot/max((ref_sq*actual_sq)**.5,1e-30),max_abs=maximum)


class Probe(TrainerCallback):
    def __init__(self,student,root,reference,rank,world,profile=False):
        self.student=student;self.root=root;self.reference=reference;self.rank=rank;self.world=world
        self.times=[];self.lrs=[];self.gradient=None;self.profile=profile;self.prof=None;self.updated_first=False

    def on_step_begin(self,args,state,control,optimizer=None,**kwargs):
        self.lrs.append(optimizer.param_groups[0]['lr'])
        if self.profile and state.global_step==3 and self.rank==0:
            self.prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False)
            self.prof.__enter__()
        torch.cuda.synchronize();self.started=time.perf_counter()

    def on_pre_optimizer_step(self,args,state,control,**kwargs):
        if state.global_step!=0:return
        torch.cuda.synchronize();self.cold_compute_seconds=time.perf_counter()-self.started
        # DDP reductions and Trainer clipping have completed. Compare ALL actual
        # first-update gradients, including tied embeddings, on rank0 only.
        if self.rank==0:
            gradients={k:p.grad.detach().float().cpu().clone() for k,p in self.student.named_parameters() if p.grad is not None}
            if len(gradients)!=sum(1 for p in self.student.parameters() if p.requires_grad):raise ValueError('Unused trainable parameter')
            if self.reference is None:
                torch.save(gradients,self.root/'first_gradients.pt');self.gradient=dict(relative_l2=0.,cosine=1.,max_abs=0.)
            else:
                expected=torch.load(self.reference/'first_gradients.pt',map_location='cpu',weights_only=True)
                self.gradient=compare_tensors(gradients,expected)
            write_json(self.root/'gradient.json',self.gradient)
        torch.distributed.barrier()

    def on_step_end(self,args,state,control,**kwargs):
        torch.cuda.synchronize();self.times.append(time.perf_counter()-self.started)
        if self.prof is not None:
            self.prof.__exit__(None,None,None)
            (self.root/'profile.txt').write_text(self.prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=60))
            self.prof.export_chrome_trace(str(self.root/'profile.json'));self.prof=None
        if self.rank==0:print('PERF_STEP',state.global_step,self.times[-1],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',choices=SUITES,default='initial')
    p.add_argument('--variant',required=True);p.add_argument('--reference',type=Path)
    p.add_argument('--steps',type=int,default=16);p.add_argument('--profile',action='store_true')
    a=p.parse_args()
    if a.steps!=16:raise ValueError('Fixed discarded 16-update probe')
    from accelerate import PartialState
    state=PartialState();rank=state.process_index;world=state.num_processes
    if state.device.type!='cuda' or world!=8:raise ValueError('Eight actual GPUs required')
    if rank==0:a.output.mkdir(parents=True,exist_ok=False)
    torch.distributed.barrier()
    option=SUITES[a.suite][a.variant];set_seed(0)
    assets=json.loads((a.inputs/'assets.json').read_text())
    student=load_parent(assets,'dclm','P300',state.device);prune(student,layers_for(FIXED_SETS['S1']))
    teacher=load_parent(assets,'dclm','P300',state.device,teacher=True)
    rows=PackedRows(a.inputs/'data','heal',count=a.steps*64)
    # Prefix identical for all variants. Per-rank assignment can change with
    # microbatch but each effective update consumes the exact same 64 sequences.
    args=make_arguments(a.output/'trainer',a.steps,3e-4,option.microbatch,world,cuda=True)
    plain=a.suite=='mbs8' or a.variant in ('reference','reference_repeat')
    model=KDModel(student,option.chunk) if plain else PerformanceKDModel(student,option)
    probe=Probe(student,a.output,a.reference,rank,world,a.profile)
    kwargs=dict(model=model,teacher=teacher,args=args,train_dataset=rows,
        data_collator=default_data_collator,callbacks=[probe])
    trainer=HealTrainer(**kwargs) if plain else PerformanceTrainer(options=option,**kwargs)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();trainer.train();elapsed=time.monotonic()-started
    peak=torch.tensor(float(torch.cuda.max_memory_allocated()),device=state.device)
    torch.distributed.all_reduce(peak,op=torch.distributed.ReduceOp.MAX)
    trace=trainer.seen if plain else trainer.order_trace();traces=[None]*world
    torch.distributed.all_gather_object(traces,trace);coverage=audit_order(traces,rows.order.tolist(),world,option.microbatch)
    if any(p.grad is not None or p.requires_grad for p in teacher.parameters()):raise ValueError('Teacher gradient')
    expected=[lr_for_update(j,a.steps,3e-4) for j in range(1,a.steps+1)]
    if not np.allclose(probe.lrs,expected,rtol=1e-12,atol=0) or trainer.state.global_step!=a.steps:raise ValueError('Schedule/steps changed')
    times=torch.tensor(probe.times,device=state.device);torch.distributed.all_reduce(times,op=torch.distributed.ReduceOp.MAX)
    # Exclude startup, expensive first-gradient audit, and profiler update4.
    steady=times.cpu().tolist()[5:]
    optimizer=trainer.optimizer;optimizer.zero_grad(set_to_none=True);del optimizer,trainer
    manifest=json.loads((a.inputs/'data/manifest.json').read_text())
    val=score(student,PackedRows(a.inputs/'data','tuning',count=len(manifest['small_indices']),order=manifest['small_indices']),distributed=True)
    if rank==0:
        initial_grad=probe.gradient
        validation_delta=0. if a.reference is None else val['loss']-json.loads((a.reference/'result.json').read_text())['validation_nll']
        accepted=(initial_grad['relative_l2']<=.02 and initial_grad['cosine']>=.9998 and abs(validation_delta)<=.002)
        result=dict(suite=a.suite,status='complete',numerical_passed=accepted,variant=a.variant,options=option.record(),
            first_gradient=initial_grad,validation_nll=val['loss'],validation_delta=validation_delta,
            thresholds=dict(gradient_relative_l2=.02,gradient_cosine=.9998,validation_abs_nats=.002),
            all_step_seconds=times.cpu().tolist(),steady_mean_seconds=statistics.mean(steady),
            steady_median_seconds=statistics.median(steady),tokens_per_second=262144/statistics.mean(steady),
            training_seconds=elapsed,cold_first_update_compute_seconds=probe.cold_compute_seconds,
            peak_allocated_bytes=peak.item(),coverage=coverage,lr_updates=probe.lrs,
            teacher_no_gradient=True,code_commit=os.environ.get('A2_CODE_COMMIT'),
            assets_sha256=hashlib.sha256((a.inputs/'assets.json').read_bytes()).hexdigest(),
            data_sha256=hashlib.sha256((a.inputs/'data/manifest.json').read_bytes()).hexdigest(),
            note='Discarded performance probe; no scientific fit. Timing excludes profiler/startup/gradient audit.')
        write_json(a.output/'result.json',result);print('PERF_RESULT',json.dumps(result),flush=True)
    torch.distributed.barrier()


if __name__=='__main__':main()
