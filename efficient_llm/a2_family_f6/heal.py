"""Registered F6 stable-state fits using unchanged A2 Trainer/KD numerics.

Scientific audits record failures and continue; no time cap or extra retry.
"""
import argparse
import gc
import json
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
from .common import Checks,read,write,SPEC_SHA
from efficient_llm.diagnostic_data import sha256 as sha

class Trace(TrainerCallback):
    def __init__(self):self.lrs=[]
    def on_step_begin(self,args,state,control,optimizer=None,**kw):self.lrs.append(float(optimizer.param_groups[0]['lr']))

class ReportTrainer(HealTrainer):
    def __init__(self,*args,trace_path,**kw):
        super().__init__(*args,**kw)
        self.trace_file=Path(trace_path).open("x",buffering=1)

    # Identical KL computation. A nonfinite value is retained/flagged rather than
    # raising the old scientific FloatingPointError and cancelling later fits.
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        ids=inputs['input_ids'];seen=inputs['sample_id'].detach().cpu().tolist();self.seen.append(seen)
        self.trace_file.write(json.dumps(dict(microbatch=len(self.seen)-1,optimizer_step=self.state.global_step,sample_ids=seen))+'\n')
        with torch.no_grad(),autocast(ids.device):hidden=self.teacher.model(ids,use_cache=False,return_dict=True).last_hidden_state
        outputs=model(input_ids=ids,teacher_hidden=hidden,teacher_weight=self.teacher.lm_head.weight)
        if not torch.isfinite(outputs['loss'].detach()):
            self.nonfinite_steps=getattr(self,'nonfinite_steps',0)+1
            print('F6_NONFINITE_KD_FLAG',flush=True)
        return (outputs['loss'],outputs) if return_outputs else outputs['loss']

def run(cfg, item):
    from accelerate import PartialState
    state=PartialState();rank=state.process_index;world=state.num_processes;device=state.device
    root=Path(cfg['results_root']);name=item['run_id'];ds=item['dataset'];arm=item['arm']
    steps=item['steps'];block='S1' if len(item['removed'])==6 else 'L'+str(len(item['removed']))
    stream=cfg['heal_streams'][ds][str(item['stream_seed'])]
    out=root/'heals'/name;initial=teacher_arm=arm;removed=item['removed']
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    state.wait_for_everyone();checks=Checks(out/f'tests-rank{rank}.json')
    checks.record('eight_ranks',world==8,dict(world=world))
    for flag in cfg.get('preparation_flags',[]):checks.record('preparation:'+flag['test'],False,flag)
    assets=read_locked(cfg['assets']);data=Path(cfg['data_root']);manifest=read(data/'manifest.json')
    ev={k:PackedRows(data,k,verify=False) for k in ('final','tuning')}
    ev['small']=PackedRows(data,'tuning',order=manifest['small_indices'],count=len(manifest['small_indices']),verify=False)
    student=load_parent(assets,ds,initial,device)
    student.config.a2_family_spec_sha256=SPEC_SHA
    baseline=score(student,ev['final'])
    ref=checks.attempt('reference_parent',lambda:read_locked(cfg['reference_nll'][f'{ds}/{initial}/parent']))
    reference=None
    if ref is not None:
        reference=ref.get('loss',ref.get('mean_nll'))
        if reference is None:reference=float(np.mean(ref['per_sequence_nll']))
    checks.record('parent_vs_A2Q',reference is not None and abs(baseline['loss']-reference)<=1e-6,dict(actual=baseline['loss'],reference=reference,difference=baseline['loss']-reference if reference is not None else None))
    with bypass(student,removed):pre={k:score(student,ev[k]) for k in ('final','tuning')}
    kept=prune(student,removed)
    checks.record('FT4_kept',kept==[i for i in range(24) if i not in removed],dict(removed=removed,kept=kept))
    teacher=load_parent(assets,ds,teacher_arm,device,teacher=True)
    rows=PackedRows(stream['path'],'heal',count=steps*64,verify=False)
    checks.record('heal_manifest',sha(Path(stream['path'])/'manifest.json')==stream['sha256'])
    set_seed(0);args=make_arguments(out/'trainer',steps,1e-3,4,world,cuda=True);trace=Trace()
    trainer=ReportTrainer(model=KDModel(student,256),teacher=teacher,args=args,train_dataset=rows,data_collator=default_data_collator,callbacks=[trace],trace_path=out/f'sample-order-rank{rank}.jsonl')
    if rank==0:write(out/'recipe.json',dict(run_id=name,dataset=ds,arm=arm,teacher=teacher_arm,initial_parent=initial,block=block,removed=removed,steps=steps,tokens=steps*64*4096,lr=.001,training=args.to_dict(),baseline=baseline,pre=pre,source=cfg['source_manifest'],registered_run=item))
    started=time.monotonic()
    try:trainer.train()
    finally:trainer.trace_file.close()
    train_seconds=time.monotonic()-started
    checks.record('FT4_updates',trainer.state.global_step==steps,trainer.state.global_step)
    expected=[lr_for_update(i,steps,.001) for i in range(1,steps+1)]
    checks.record('FT4_LR',len(trace.lrs)==steps and np.allclose(trace.lrs,expected,rtol=1e-12,atol=0),dict(updates=len(trace.lrs),first=trace.lrs[:2],last=trace.lrs[-2:]))
    traces=[None]*world;torch.distributed.all_gather_object(traces,trainer.seen)
    coverage=checks.attempt('FT4_data_order',lambda:audit_order(traces,rows.order.tolist(),world,4))
    if coverage is not None:checks.record('FT4_data_order',True,coverage)
    checks.record('FT4_teacher_frozen',not any(p.grad is not None or p.requires_grad for p in teacher.parameters()))
    checks.record('finite_training',getattr(trainer,'nonfinite_steps',0)==0,dict(nonfinite_microbatches=getattr(trainer,'nonfinite_steps',0)))
    history=trainer.state.log_history
    trainer.optimizer.zero_grad(set_to_none=True);del trainer;gc.collect();torch.cuda.empty_cache()
    master=checks.attempt('master_final',lambda:score(student,ev['final']))
    round_to_saved_precision(student)
    # Preserve checkpoint even if a numerical scoring diagnostic fails.
    if rank==0:checks.attempt('save_student',lambda:save_student(student,out/'final'))
    state.wait_for_everyone()
    values={k:checks.attempt('post:'+k,lambda k=k:score(student,ev[k],teacher=teacher if k=='small' else None)) for k in ev}
    if item['phase']=='FT9' and values['final'] is not None:
        checks.record('FT9',abs(values['final']['loss']-cfg['ft9_reference'])<=.002,dict(actual=values['final']['loss'],reference=cfg['ft9_reference'],tolerance=.002,origin='A2 S on 6y3o'))
    if values['tuning'] is not None:checks.record('tuning_divergence',values['tuning']['loss']<=pre['tuning']['loss'],dict(pre=pre['tuning']['loss'],post=values['tuning']['loss']))
    reloaded=checks.attempt('reload_model',lambda:reload_student(out/'final',device))
    reload_ok=torch.tensor(int(reloaded is not None),device=device)
    torch.distributed.all_reduce(reload_ok,op=torch.distributed.ReduceOp.MIN)
    checks.record('all_rank_reload_available',bool(reload_ok.item()))
    reload_value=checks.attempt('reload_score',lambda:score(reloaded,ev['small'])) if reload_ok.item() else None
    difference=abs(reload_value['loss']-values['small']['loss']) if reload_value and values['small'] else None
    checks.record('FT4_roundtrip',difference is not None and difference<=1e-6,dict(difference=difference))
    all_checks=[None]*world
    torch.distributed.all_gather_object(all_checks,checks.items)
    flags=[dict(c,rank=i) for i,items in enumerate(all_checks) for c in items if not c['passed']]
    if rank==0:
        write(out/'order.json',dict(rank_microbatches=traces))
        result=dict(run_id=name,dataset=ds,arm=arm,block=block,initial_parent=initial,teacher=teacher_arm,status='complete',training_seconds=train_seconds,elapsed_seconds=time.monotonic()-started,baseline=baseline,pre=pre,scores=values,master_final=master,lr_updates=trace.lrs,coverage=coverage,roundtrip_difference=difference,checkpoint=str(out/'final'),delta_post=values['final']['loss']-baseline['loss'] if values['final'] else None)
        result.update(raw_order_files=[dict(path=str(out/f'sample-order-rank{i}.jsonl'),sha256=sha(out/f'sample-order-rank{i}.jsonl')) for i in range(world)],registered_run=item,flags=flags,unverified=bool(flags),log_history=history)
        write(out/'result.json',result)
        for label,value in [('parent',baseline),('pre-'+block,pre['final']),('post-'+block,values['final'])]:
            if value is not None:write(root/'perseq'/name/(label+'.json'),dict(value,run_id=name,flags=flags,unverified=bool(flags)))
    state.wait_for_everyone()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args();cfg=read(a.config)
    from .protocol import descriptor
    item=cfg['runs'][a.run_id]
    if item!=descriptor(item['dataset'],item['arm'],item['budget']):raise ValueError('Unregistered F6 recipe')
    if cfg['spec_sha256']!=SPEC_SHA:raise ValueError('Wrong F6 spec')
    run(cfg,item)
