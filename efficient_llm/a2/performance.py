"""Opt-in A2 performance candidates. Historical model/train paths are unchanged.

Only the loss chunk may be compiled; the decoder, attention and teacher identity
remain native. Deferred checks still abort BEFORE every optimizer update.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import TrainerCallback
from .model import kl_chunk, autocast
from .train import HealTrainer


@dataclass(frozen=True)
class Options:
    microbatch: int = 1
    chunk: int = 128
    deferred_checks: bool = False
    fused_optimizer: bool = False
    compile_head: bool = False

    def __post_init__(self):
        if self.microbatch not in (1,2,4) or self.chunk not in (128,256,512,1024):
            raise ValueError('Unreviewed performance configuration')

    def record(self):return asdict(self)


VARIANTS = {
    'reference': Options(),
    'deferred': Options(deferred_checks=True),
    'fused': Options(fused_optimizer=True),
    'chunk512': Options(chunk=512),
    'chunk1024': Options(chunk=1024),
    'micro2': Options(microbatch=2,chunk=512),
    'micro4': Options(microbatch=4,chunk=256),
    'combined': Options(chunk=512,deferred_checks=True,fused_optimizer=True),
    'compiled_head': Options(chunk=512,deferred_checks=True,fused_optimizer=True,compile_head=True),
    'reference_repeat': Options(),
}


class PerformanceKDModel(nn.Module):
    def __init__(self,student,options,*,compile_backend='inductor'):
        super().__init__();self.student=student;self.config=student.config;self.options=options
        self.loss_chunk=kl_chunk
        if options.compile_head:
            # HF backward runs outside the forward autocast context. Be explicit:
            # compiled backward must not assume it is inside that context.
            import torch._functorch.config
            torch._functorch.config.backward_pass_autocast='off'
            self.loss_chunk=torch.compile(kl_chunk,backend=compile_backend,fullgraph=True,dynamic=False)

    def forward(self,input_ids,teacher_hidden,teacher_weight):
        h=self.student.model(input_ids,use_cache=False,return_dict=True).last_hidden_state
        loss=h.new_zeros((),dtype=torch.float32)
        for start in range(0,h.shape[1],self.options.chunk):
            args=(h[:,start:start+self.options.chunk],self.student.lm_head.weight,
                  teacher_hidden[:,start:start+self.options.chunk],teacher_weight)
            loss=loss+checkpoint(self.loss_chunk,*args,use_reentrant=False,preserve_rng_state=False)
        return {'loss':loss/(h.shape[0]*h.shape[1])}


class FiniteBeforeUpdate(TrainerCallback):
    def __init__(self,trainer):self.trainer=trainer
    def on_pre_optimizer_step(self,*args,**kwargs):
        trainer=self.trainer
        if not trainer.pending_finite:raise RuntimeError('No loss checked for update')
        good=torch.stack(trainer.pending_finite).all().to(dtype=torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(good,op=torch.distributed.ReduceOp.MIN)
        trainer.pending_finite.clear()
        if not bool(good):raise FloatingPointError('Nonfinite KD loss before optimizer update')


class PerformanceTrainer(HealTrainer):
    def __init__(self,*args,options,**kwargs):
        self.options=options;self.pending_finite=[];self.device_ids=[]
        super().__init__(*args,**kwargs)
        if options.deferred_checks:self.add_callback(FiniteBeforeUpdate(self))

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        if not self.options.deferred_checks:
            return super().compute_loss(model,inputs,return_outputs,num_items_in_batch)
        ids=inputs['input_ids'];self.device_ids.append(inputs['sample_id'].detach())
        with torch.no_grad(),autocast(ids.device):
            hidden=self.teacher.model(ids,use_cache=False,return_dict=True).last_hidden_state
        outputs=model(input_ids=ids,teacher_hidden=hidden,teacher_weight=self.teacher.lm_head.weight)
        loss=outputs['loss'];self.pending_finite.append(torch.isfinite(loss.detach()))
        return (loss,outputs) if return_outputs else loss

    def order_trace(self):
        if self.pending_finite:raise RuntimeError('Unchecked loss remains')
        if not self.options.deferred_checks:return self.seen
        if not self.device_ids:return []
        return torch.stack(self.device_ids).cpu().tolist()

    def create_optimizer(self):
        if not self.options.fused_optimizer:return super().create_optimizer()
        if self.optimizer is None:
            decay=[];no_decay=[]
            for p in self.model.parameters():
                if p.requires_grad:(decay if p.ndim>=2 else no_decay).append(p)
            self.optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':.1},
                {'params':no_decay,'weight_decay':0.}],lr=self.args.learning_rate,
                betas=(.9,.95),eps=1e-8,fused=True)
        return self.optimizer
