"""Native HF Trainer healing. New optimizer/full schedule for EACH budget."""
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainingArguments, TrainerCallback, default_data_collator, set_seed
from .model import KDModel, autocast, save_student, round_to_saved_precision, reload_student
from .measure import score, rank_world
from .data import write_json
from .protocol import lr_for_update


class HealTrainer(Trainer):
    def __init__(self,*args,teacher,**kwargs):
        super().__init__(*args,**kwargs)
        self.teacher=teacher.eval().requires_grad_(False)
        self.model_accepts_loss_kwargs=False
        self.seen=[]
        if self.args.gradient_checkpointing: raise ValueError('No decoder recomputation in A2')
        if self.args.label_smoothing_factor: raise ValueError('Pure forward KL only')

    def _get_train_sampler(self,train_dataset=None):
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        ids=inputs['input_ids']; self.seen.append(inputs['sample_id'].detach().cpu().tolist())
        with torch.no_grad(),autocast(ids.device):
            hidden=self.teacher.model(ids,use_cache=False,return_dict=True).last_hidden_state
        outputs=model(input_ids=ids,teacher_hidden=hidden,teacher_weight=self.teacher.lm_head.weight)
        loss=outputs['loss']
        if not torch.isfinite(loss.detach()): raise FloatingPointError('Nonfinite KD loss')
        return (loss,outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is None:
            decay,no_decay=[],[]
            for p in self.model.parameters():
                if p.requires_grad: (decay if p.ndim>=2 else no_decay).append(p)
            self.optimizer=torch.optim.AdamW([
                {'params':decay,'weight_decay':.1}, {'params':no_decay,'weight_decay':0.}],
                lr=self.args.learning_rate,betas=(.9,.95),eps=1e-8)
        return self.optimizer

    def create_scheduler(self,num_training_steps,optimizer=None):
        if self.lr_scheduler is None:
            # LambdaLR is initialized at index 0 BEFORE update 1; HF's default
            # warmup starts at zero and is not the preregistered update convention.
            self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer,
                lambda completed: lr_for_update(min(completed+1,num_training_steps),num_training_steps,1.))
            self._created_lr_scheduler=True
        return self.lr_scheduler


class RuntimeGuard(TrainerCallback):
    def __init__(self,estimated_seconds):
        if not math.isfinite(estimated_seconds) or estimated_seconds<=0: raise ValueError('Positive measured runtime estimate required')
        self.limit=1.5*estimated_seconds; self.lrs=[]; self.started=None
    def on_train_begin(self,args,state,control,**kwargs): self.started=time.monotonic()
    def on_step_begin(self,args,state,control,optimizer=None,**kwargs):
        if time.monotonic()-self.started>self.limit: raise TimeoutError('Run exceeded 1.5x measured estimate')
        self.lrs.append(float(optimizer.param_groups[0]['lr']))


def audit_order(batches,expected,world,microbatch):
    """Reconstruct global order from actual per-rank microbatch traces."""
    if len(batches)!=world or len({len(b) for b in batches})!=1:
        raise ValueError('Ranks consumed different numbers of microbatches')
    joined=[]
    for step in range(len(batches[0])):
        for rank in range(world):
            row=batches[rank][step]
            if len(row)!=microbatch: raise ValueError('Incomplete microbatch')
            joined.extend(row)
    if joined != list(expected): raise ValueError('Actual global sample order/coverage differs')
    return dict(examples=len(joined),microbatches=len(batches[0]),world_size=world,exact_order=True)


def make_arguments(output,steps,lr,microbatch,world,*,cuda,global_sequences=64):
    if global_sequences%(world*microbatch): raise ValueError('Microbatch/world must divide global sequences')
    return TrainingArguments(output_dir=str(output),max_steps=steps,
        learning_rate=lr,per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=global_sequences//(world*microbatch),
        optim='adamw_torch',adam_beta1=.9,adam_beta2=.95,weight_decay=.1,max_grad_norm=1.,
        bf16=cuda,use_cpu=not cuda,seed=0,data_seed=0,gradient_checkpointing=False,
        dataloader_drop_last=True,dataloader_num_workers=0,dataloader_pin_memory=cuda,
        report_to='none',save_strategy='no',eval_strategy='no',logging_steps=10,
        logging_first_step=True,logging_nan_inf_filter=False,disable_tqdm=True,
        remove_unused_columns=False,ddp_find_unused_parameters=False,ddp_broadcast_buffers=False,
        accelerator_config={'split_batches':False,'dispatch_batches':False,'even_batches':False},
        label_names=[])


def heal(student,teacher,train_rows,evaluation,output,*,run,microbatch=1,chunk=128,
         estimated_seconds,provenance,pre_values,global_sequences=64,rerun_of=None):
    """One fresh run, no implicit resume/retry. Caller records failed run separately."""
    rank,world=rank_world(); root=Path(output)
    if rank==0: root.mkdir(parents=True,exist_ok=False)
    if world>1: torch.distributed.barrier()
    if run.lr is None or len(train_rows)!=run.steps*global_sequences:
        raise ValueError('Resolved LR and exact run prefix required')
    if 'secondary' in evaluation and (not pre_values.get('secondary') or
        any(pre_values.get(k) is None for k in ('parent_secondary','teacher_secondary'))):
        raise ValueError('Secondary scoring requires matching unhealed/parent/teacher baselines')
    device=next(student.parameters()).device
    if any(p.dtype!=torch.float32 for p in student.parameters()): raise ValueError('FP32 student required')
    set_seed(0)
    args=make_arguments(root/'trainer',run.steps,run.lr,microbatch,world,
                        cuda=device.type=='cuda',global_sequences=global_sequences)
    guard=RuntimeGuard(estimated_seconds)
    trainer=HealTrainer(model=KDModel(student,chunk),teacher=teacher,args=args,
        train_dataset=train_rows,data_collator=default_data_collator,callbacks=[guard])
    started=time.monotonic()
    if rank==0: write_json(root/'recipe.json',dict(run.record(),provenance=provenance,
        training=args.to_dict(),estimated_seconds=estimated_seconds))
    try:
        trainer.train()
    except FloatingPointError as error:
        result=dict(run.record(),status='failed',diverged=True,error=str(error),provenance=provenance,rerun_of=rerun_of)
        if rank==0: write_json(root/'result.json',result)
        raise
    training_seconds=time.monotonic()-started
    if trainer.state.global_step!=run.steps: raise ValueError('Wrong number of updates')
    expected_lrs=[lr_for_update(j,run.steps,run.lr) for j in range(1,run.steps+1)]
    if not np.allclose(guard.lrs,expected_lrs,rtol=1e-12,atol=0): raise ValueError('LR update sequence mismatch')
    traces=[None]*world
    if world>1: torch.distributed.all_gather_object(traces,trainer.seen)
    else: traces[0]=trainer.seen
    coverage=audit_order(traces,train_rows.order.tolist(),world,microbatch)
    if any(p.grad is not None or p.requires_grad for p in teacher.parameters()): raise ValueError('Teacher gradient')
    # Release optimizer and DDP wrapper before scoring/exporting; keep native model.
    trainer.optimizer.zero_grad(set_to_none=True)
    history=trainer.state.log_history
    del trainer
    try:
        master=score(student,evaluation['final'])['loss']
        round_to_saved_precision(student)
        values={key:score(student,rows,teacher=teacher if key=='small' else None)
                for key,rows in evaluation.items()}
    except FloatingPointError as error:
        result=dict(run.record(),status='failed',diverged=True,error=str(error),provenance=provenance,rerun_of=rerun_of)
        if rank==0:write_json(root/'result.json',result)
        raise
    diverged=values['tuning']['loss']>pre_values['tuning']['loss']
    # Save first, then score the actual exported file for the round-trip gate.
    if rank==0: save_student(student,root/'final')
    if world>1: torch.distributed.barrier()
    reloaded=reload_student(root/'final',device)
    reload_val=score(reloaded,evaluation['small'])['loss']
    difference=abs(reload_val-values['small']['loss']); del reloaded
    if difference>1e-6: raise ValueError('T16 BF16 saved checkpoint score differs')
    elapsed=time.monotonic()-started
    result=dict(run.record(),status='complete',diverged=diverged,provenance=provenance,rerun_of=rerun_of,
        val_parent=pre_values['parent_final'],val_teacher=pre_values['teacher_final'],
        val_pre=pre_values['final']['loss'],val_post=values['final']['loss'],
        delta_pre=pre_values['final']['loss']-pre_values['parent_final'],
        delta_post=values['final']['loss']-pre_values['parent_final'],
        val_tune_pre=pre_values['tuning']['loss'],val_tune_post=values['tuning']['loss'],
        val_post_master=master,final_kl=values['small']['final_kl'],scores=values,
        coverage=coverage,roundtrip_difference=difference,training_seconds=training_seconds,
        wallclock_h=elapsed/3600,gpu_hours=elapsed*world/3600,
        tokens_per_second=run.steps*global_sequences*train_rows.length/training_seconds,
        lr_updates=guard.lrs,log_history=history)
    if 'secondary' in values:
        result.update(val_secondary_parent=pre_values['parent_secondary'],
            val_secondary_teacher=pre_values['teacher_secondary'],
            val_secondary_pre=pre_values['secondary']['loss'],
            val_secondary_post=values['secondary']['loss'],
            delta_secondary_pre=pre_values['secondary']['loss']-pre_values['parent_secondary'],
            delta_secondary_post=values['secondary']['loss']-pre_values['parent_secondary'])
    if rank==0:
        write_json(root/'order.json',traces); write_json(root/'result.json',result)
    return result
