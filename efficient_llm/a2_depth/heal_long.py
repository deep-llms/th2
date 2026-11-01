"""Registered Depth Part C: full 12.8B schedule with native resumable state.

Scientific audits record failures and continue; no time cap or extra retry.
"""
import argparse
import gc
import time
from pathlib import Path
import numpy as np
import torch
from transformers import TrainerCallback,default_data_collator,set_seed
from efficient_llm.a2.train import HealTrainer,make_arguments,audit_order
from efficient_llm.a2.model import load_parent,prune,bypass,KDModel,autocast,round_to_saved_precision,save_student,reload_student
from efficient_llm.a2.measure import score
from efficient_llm.a2.protocol import lr_for_update
from efficient_llm.a2.data import PackedRows
from scripts.a2_remaining import read_locked
from .common import Checks,read,write
from efficient_llm.diagnostic_data import sha256 as sha

from .heal_short import Trace, ReportTrainer
from .resume import enable_checkpoints, CheckpointAudit, select_checkpoint, restore_audit
from transformers.trainer_callback import TrainerState

class RecoverableTrainer(ReportTrainer):
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        value=super().compute_loss(model,inputs,return_outputs,num_items_in_batch)
        loss=value[0] if return_outputs else value
        bad=(~torch.isfinite(loss.detach())).to(dtype=torch.int32)
        if torch.distributed.is_initialized():torch.distributed.all_reduce(bad,op=torch.distributed.ReduceOp.MAX)
        if bad.item():
            # All ranks leave the failed attempt together, before a corrupt update.
            # The coordinator records it and resumes the full registered schedule.
            self.recovery_checks.record('nonfinite_attempt',False,dict(step=self.state.global_step,action='recover clean state'))
            print('DEPTH_NONFINITE_RECOVERY',self.state.global_step,flush=True)
            raise FloatingPointError('Recoverable nonfinite KD; no optimizer update applied')
        return value


STEPS=48828

def run_identity(cfg, arm):
    return dict(spec_sha256=cfg['spec_sha256'], steps=STEPS, lr=.001, global_sequences=64,
                microbatch=4, dataset='dclm', arm=arm, removed=list(range(16,22)),
                assets=cfg['assets'], heal_stream=cfg['heal_streams']['dclm'],
                source_manifest=cfg['source_manifest'])

def train_resumable(trainer, trace, identity, checkpoint, out, *, resume=False, rejected=()):
    """Use the native state loader; completed checkpoints need evaluation only."""
    trainer.add_callback(CheckpointAudit(trainer,trace,identity))
    if checkpoint:restore_audit(trainer,trace,checkpoint)
    if trainer.args.process_index==0:
        events=read(out/'resume_events.json').get('events',[]) if (out/'resume_events.json').exists() else []
        events.append(dict(attempt=len(events)+1,resume_requested=resume,checkpoint=str(checkpoint) if checkpoint else None,
                           rejected=list(rejected),action='resume' if checkpoint else 'full schedule from initialization'))
        write(out/'resume_events.json',dict(events=events))
    if checkpoint and read(checkpoint/'trainer_state.json')['global_step']==identity['steps']:
        trainer._load_from_checkpoint(str(checkpoint))
        trainer.state=TrainerState.load_from_json(str(checkpoint/'trainer_state.json'))
    else:
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)


def run(cfg,arm,resume=False):
    ds,block="dclm","S1"
    from accelerate import PartialState
    state=PartialState();rank=state.process_index;world=state.num_processes;device=state.device
    root=Path(cfg['results_root'])
    name=f'ST1-dclm-{arm}-S1-12.8B-lr1e-3'
    out=root/'heals'/name
    initial='S72' if arm=='X72' else arm
    teacher_arm='P300' if arm=='X72' else arm
    removed=list(range(16,22))
    if rank==0:out.mkdir(parents=True,exist_ok=resume)
    state.wait_for_everyone()
    # Preserve earlier attempt checks before the new Checks writer starts.
    prior_path=out/f'tests-rank{rank}.json'
    prior=read(prior_path).get('checks',[]) if resume and prior_path.exists() else []
    checks=Checks(prior_path)
    for item in prior:checks.record(item['test'],item['passed'],item.get('detail'))
    identity=run_identity(cfg,arm)
    checkpoint,rejected=select_checkpoint(out/'trainer',identity,world) if resume else (None,[])
    for item in rejected:checks.record('resume_rejected_state',False,item)
    if resume:checks.record('resume_available',checkpoint is not None,dict(checkpoint=str(checkpoint),action='resume matching complete state; if absent preserve flags and run full registered schedule'))
    checks.record('eight_ranks',world==8,dict(world=world))
    assets=read_locked(cfg['assets']);data=Path(cfg['data_root']);manifest=read(data/'manifest.json')
    ev={k:PackedRows(data,k,verify=False) for k in ('final','tuning')}
    ev['small']=PackedRows(data,'tuning',order=manifest['small_indices'],count=len(manifest['small_indices']),verify=False)
    student=load_parent(assets,ds,initial,device)
    baseline=score(student,ev['final'])
    ref=read_locked(cfg['reference_nll'][f'{ds}/{initial}/parent'])
    reference=ref.get('loss',ref.get('mean_nll'))
    if reference is None:reference=float(np.mean(ref['per_sequence_nll']))
    checks.record('parent_vs_A2Q',abs(baseline['loss']-reference)<=1e-6,dict(actual=baseline['loss'],reference=reference,difference=baseline['loss']-reference))
    with bypass(student,removed):pre={k:score(student,ev[k]) for k in ('final','tuning')}
    kept=prune(student,removed)
    checks.record('DT2',kept==[i for i in range(24) if i not in removed],dict(removed=removed,kept=kept))
    teacher=load_parent(assets,ds,teacher_arm,device,teacher=True)
    rows=PackedRows(cfg['heal_streams'][ds]['path'],'heal',count=STEPS*64,verify=False)
    checks.record('heal_manifest',sha(Path(cfg['heal_streams'][ds]['path'])/'manifest.json')==cfg['heal_streams'][ds]['sha256'])
    set_seed(0);args=enable_checkpoints(make_arguments(out/'trainer',STEPS,1e-3,4,world,cuda=True));trace=Trace()
    trainer=RecoverableTrainer(model=KDModel(student,256),teacher=teacher,args=args,train_dataset=rows,data_collator=default_data_collator,callbacks=[trace])
    trainer.recovery_checks=checks
    if rank==0 and not (out/'recipe.json').exists():write(out/'recipe.json',dict(run_id=name,dataset=ds,arm=arm,teacher=teacher_arm,initial_parent=initial,block=block,removed=removed,steps=STEPS,tokens=STEPS*64*4096,lr=.001,training=args.to_dict(),baseline=baseline,pre=pre,source=cfg['source_manifest']))
    started=time.monotonic()
    train_resumable(trainer,trace,identity,checkpoint,out,resume=resume,rejected=rejected)
    train_seconds=time.monotonic()-started
    checks.record('DT5_updates',trainer.state.global_step==STEPS,trainer.state.global_step)
    expected=[lr_for_update(i,STEPS,.001) for i in range(1,STEPS+1)]
    checks.record('DT5_LR',len(trace.lrs)==STEPS and np.allclose(trace.lrs,expected,rtol=1e-12,atol=0),dict(updates=len(trace.lrs),first=trace.lrs[:2],last=trace.lrs[-2:]))
    traces=[None]*world;torch.distributed.all_gather_object(traces,trainer.seen)
    coverage=checks.attempt('DT5_data_order',lambda:audit_order(traces,rows.order.tolist(),world,4))
    if coverage is not None:checks.record('DT5_data_order',True,coverage)
    checks.record('DT5_teacher_frozen',not any(p.grad is not None or p.requires_grad for p in teacher.parameters()))
    checks.record('finite_training',getattr(trainer,'nonfinite_steps',0)==0,dict(nonfinite_microbatches=getattr(trainer,'nonfinite_steps',0)))
    history=trainer.state.log_history
    if trainer.optimizer is not None:trainer.optimizer.zero_grad(set_to_none=True)
    del trainer;gc.collect();torch.cuda.empty_cache()
    master=checks.attempt('master_final',lambda:score(student,ev['final']))
    round_to_saved_precision(student)
    # Preserve checkpoint even if a numerical scoring diagnostic fails.
    if rank==0:save_student(student,out/'final')
    state.wait_for_everyone()
    values={k:checks.attempt('post:'+k,lambda k=k:score(student,ev[k],teacher=teacher if k=='small' else None)) for k in ev}
    if values['tuning'] is not None:checks.record('tuning_divergence',values['tuning']['loss']<=pre['tuning']['loss'],dict(pre=pre['tuning']['loss'],post=values['tuning']['loss']))
    reloaded=reload_student(out/'final',device);reload_value=checks.attempt('reload_score',lambda:score(reloaded,ev['small']))
    difference=abs(reload_value['loss']-values['small']['loss']) if reload_value and values['small'] else None
    checks.record('DT5_roundtrip',difference is not None and difference<=1e-6,dict(difference=difference))
    all_checks=[None]*world
    torch.distributed.all_gather_object(all_checks,checks.items)
    flags=[dict(c,rank=i) for i,items in enumerate(all_checks) for c in items if not c['passed']]
    if rank==0:
        write(out/'order.json',dict(rank_microbatches=traces))
        result=dict(run_id=name,dataset=ds,arm=arm,block=block,initial_parent=initial,teacher=teacher_arm,status='complete',training_seconds=train_seconds,elapsed_seconds=time.monotonic()-started,baseline=baseline,pre=pre,scores=values,master_final=master,lr_updates=trace.lrs,coverage=coverage,roundtrip_difference=difference,checkpoint=str(out/'final'),delta_post=values['final']['loss']-baseline['loss'] if values['final'] else None)
        result.update(flags=flags,unverified=bool(flags),log_history=history,resume_events=read(out/'resume_events.json')['events'],full_state_identity=identity)
        write(out/'result.json',result)
        for label,value in [('parent',baseline),('pre-'+block,pre['final']),('post-'+block,values['final'])]:
            if value is not None:write(root/'perseq'/name/(label+'.json'),dict(value,run_id=name,flags=flags,unverified=bool(flags)))
        from .recovery import seal_arm
        seal_arm(out,identity)
    state.wait_for_everyone()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--arm',choices=('P300','P1T','X72'),required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(read(a.config),a.arm,a.resume)
