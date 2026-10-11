"""Profile one real HF Trainer update, then continue its disposable smoke.

All training arguments are forwarded unchanged to train.py. Instrumentation is
confined to this process; ordinary training never imports this module.
"""
from contextlib import nullcontext
import os
import time
from pathlib import Path
import runpy

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import TrainerCallback

from deep_kv.proxy import ProxyModel
from deep_kv.proxy_estimators import AnticipatoryHead, _MLPPath
from deep_kv.proxy_memory import MemoryHead, SimpleMemoryHead
import deep_kv.proxy_memory as memory
import deep_kv.proxy_training as training
from scripts.check_fa4_proxy import write

ACTIVE=False
PREFIX='p4_profile/'


def annotate(function,name):
    def wrapped(*args,**kwargs):
        with record_function(PREFIX+name) if ACTIVE else nullcontext():
            return function(*args,**kwargs)
    return wrapped


def summarize(trace):
    events=trace.events();sequences={};totals={}
    def owner(event):
        node=event
        while node is not None:
            if node.name.startswith(PREFIX):return node.name[len(PREFIX):]
            node=node.cpu_parent
        return None
    for event in events:
        category=owner(event)
        if category and event.sequence_nr>=0:
            sequences[(event.thread,event.sequence_nr)]=category
    for event in events:
        # CPU launch records contain their own correlated kernel duration.
        # Skip raw CUDA events to avoid counting kernels twice.
        if event.device_type!=torch.autograd.DeviceType.CPU:continue
        category=owner(event)
        if category is None:
            node=event
            while node is not None:
                category=sequences.get((getattr(node,'fwd_thread',None),node.sequence_nr))
                if category is not None:
                    category+='_backward';break
                node=node.cpu_parent
        category=category or 'remaining_optimizer_communication_and_model'
        totals[category]=totals.get(category,0.)+event.self_device_time_total/1000
    return {k:v for k,v in totals.items() if v>0}


class Capture(TrainerCallback):
    def __init__(self,trainer):
        self.trainer=trainer
        self.steps=[]

    def on_step_begin(self,args,state,control,**kwargs):
        global ACTIVE
        # Includes the preceding update's callbacks/logging/data delivery. The
        # final callback timer below measures just the complete update itself.
        torch.cuda.synchronize(args.device)
        now=time.perf_counter()
        if hasattr(self,'last_begin'):
            self.steps[-1]['interval_seconds']=now-self.last_begin
        self.last_begin=now
        if state.global_step==int(os.environ.get('P6_PROFILE_UPDATE','6'))-1 and args.process_index==0:
            torch.cuda.synchronize();self.trace=profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA])
            self.trace.__enter__();ACTIVE=True

    def on_step_end(self,args,state,control,**kwargs):
        global ACTIVE
        if args.process_index==0:
            callback=next(c for c in self.trainer.callback_handler.callbacks if isinstance(c,training.ProxyCallback))
        torch.cuda.synchronize(args.device)
        self.steps.append(dict(step=state.global_step,seconds=time.perf_counter()-self.last_begin,
                              legacy_seconds=callback.seconds_per_update if args.process_index==0 else None))
        if state.global_step==int(os.environ.get('P6_PROFILE_UPDATE','6')) and args.process_index==0:
            torch.cuda.synchronize();ACTIVE=False;self.trace.__exit__(None,None,None)
            path=Path(args.output_dir)/'component-profile.json'
            times=summarize(self.trace)
            write(path,dict(status='profiled',step=6,rank=0,world_size=args.world_size,
                kernel_milliseconds=times,auxiliary_recomputation=False,
                cpu_events=[dict(name=e.key,count=e.count,self_cpu_ms=e.self_cpu_time_total/1000,
                                 cpu_ms=e.cpu_time_total/1000,device_ms=e.device_time_total/1000)
                            for e in self.trace.key_averages()
                            if e.key.startswith(PREFIX) or any(s in e.key.lower() for s in
                                ('synchroniz','item','local_scalar','nonzero','memcpy'))],
                note='Exclusive summed kernel durations, not critical-path wall time. '
                     'Target/mask/setup includes incremental target sums, moment accumulation, '
                     'mask creation and RoPE outside decoder blocks. '
                     'Use unprofiled steps 10-25 for end-to-end throughput.'))
            self.trace=None

    def on_train_end(self,args,state,control,**kwargs):
        write(Path(args.output_dir)/f'step-profile-rank{args.process_index}.json',dict(status='profiled',steps=self.steps,
            peak_gib=torch.cuda.max_memory_allocated()/2**30,
            note='Complete callback update and start-to-start intervals on every rank. '
                 'Exclude profiled update 6 and its following interval; compilation/startup/eval/save excluded.'))
        if args.process_index==0:
            write(Path(args.output_dir)/'step-profile.json',dict(status='profiled',steps=self.steps))


def main():
    assert int(os.environ.get('WORLD_SIZE','1'))==8
    for cls,name,label in [
        (ProxyModel,'_run_backbone','target_mask_and_statistics_setup'),
        (ProxyModel,'block','remaining_decoder'),
        (ProxyModel,'attention','attention'),
        (ProxyModel,'normalize_target','target_normalization'),
        (ProxyModel,'layer_cosines','auxiliary_cosine'),
        (ProxyModel,'update_statistics','normalization_update'),
        (ProxyModel,'lm_statistics','lm_loss'),
        (AnticipatoryHead,'routed','estimator_forward'),
        (MemoryHead,'estimate','estimator_forward'),
        (SimpleMemoryHead,'estimate','estimator_forward'),
        (MemoryHead,'entries','proxy_projections'),
        (ProxyModel,'memory_attention','proxy_memory_attention')]:
        setattr(cls,name,annotate(getattr(cls,name),label))
    _MLPPath.backward=staticmethod(annotate(_MLPPath.backward,'estimator_backward'))
    memory.relational_losses=annotate(memory.relational_losses,'auxiliary_relational')
    training.reduce_moments=annotate(training.reduce_moments,'statistics_all_reduce')
    training.ProxyTrainer.training_step=annotate(training.ProxyTrainer.training_step,'training_microbatch')
    training.ProxyTrainer.compute_loss=annotate(training.ProxyTrainer.compute_loss,'forward_and_loss')
    training.ProxyCallback.on_step_end=annotate(training.ProxyCallback.on_step_end,'complete_statistics_callback')
    from deep_kv.training import PilotCallback
    PilotCallback.on_step_end=annotate(PilotCallback.on_step_end,'diagnostic_reduce_and_cpu')
    original=training.ProxyTrainer.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs);self.add_callback(Capture(self))
    training.ProxyTrainer.__init__=init
    runpy.run_module('train',run_name='__main__')


if __name__=='__main__':main()
