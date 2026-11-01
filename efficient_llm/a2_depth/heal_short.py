"""Registered Depth DT4 and twelve Part B fits using unchanged A2 Trainer/KD numerics.

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

class Trace(TrainerCallback):
    def __init__(self):self.lrs=[]
    def on_step_begin(self,args,state,control,optimizer=None,**kw):self.lrs.append(float(optimizer.param_groups[0]['lr']))

class ReportTrainer(HealTrainer):
    # Identical KL computation. A nonfinite value is retained/flagged rather than
    # raising the old scientific FloatingPointError and cancelling later fits.
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        ids=inputs['input_ids'];self.seen.append(inputs['sample_id'].detach().cpu().tolist())
        with torch.no_grad(),autocast(ids.device):hidden=self.teacher.model(ids,use_cache=False,return_dict=True).last_hidden_state
        outputs=model(input_ids=ids,teacher_hidden=hidden,teacher_weight=self.teacher.lm_head.weight)
        if not torch.isfinite(outputs['loss'].detach()):
            self.nonfinite_steps=getattr(self,'nonfinite_steps',0)+1
            print('DEPTH_NONFINITE_KD_FLAG',flush=True)
        return (outputs['loss'],outputs) if return_outputs else outputs['loss']

def run(cfg,ds,arm,block,dt4=False):
    from accelerate import PartialState
    state=PartialState();rank=state.process_index;world=state.num_processes;device=state.device
    root=Path(cfg['results_root'])
    name='ST1-dclm-P300-S1-0.2B-lr1e-3' if dt4 else f'D-{ds}-{arm}-{block}-0.2B-lr1e-3'
    out=root/'heals'/name
    initial='S72' if arm=='X72' else arm
    teacher_arm='P300' if arm=='X72' else arm
    removed=list(range(16,22)) if dt4 else cfg['layer_sets'][ds][block]
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    state.wait_for_everyone();checks=Checks(out/f'tests-rank{rank}.json')
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
    rows=PackedRows(cfg['heal_streams'][ds]['path'],'heal',count=762*64,verify=False)
    checks.record('heal_manifest',sha(Path(cfg['heal_streams'][ds]['path'])/'manifest.json')==cfg['heal_streams'][ds]['sha256'])
    set_seed(0);args=make_arguments(out/'trainer',762,1e-3,4,world,cuda=True);trace=Trace()
    trainer=ReportTrainer(model=KDModel(student,256),teacher=teacher,args=args,train_dataset=rows,data_collator=default_data_collator,callbacks=[trace])
    if rank==0:write(out/'recipe.json',dict(run_id=name,dataset=ds,arm=arm,teacher=teacher_arm,initial_parent=initial,block=block,removed=removed,steps=762,tokens=762*64*4096,lr=.001,training=args.to_dict(),baseline=baseline,pre=pre,source=cfg['source_manifest']))
    started=time.monotonic();trainer.train();train_seconds=time.monotonic()-started
    checks.record('DT5_updates',trainer.state.global_step==762,trainer.state.global_step)
    expected=[lr_for_update(i,762,.001) for i in range(1,763)]
    checks.record('DT5_LR',len(trace.lrs)==762 and np.allclose(trace.lrs,expected,rtol=1e-12,atol=0),dict(updates=len(trace.lrs),first=trace.lrs[:2],last=trace.lrs[-2:]))
    traces=[None]*world;torch.distributed.all_gather_object(traces,trainer.seen)
    coverage=checks.attempt('DT5_data_order',lambda:audit_order(traces,rows.order.tolist(),world,4))
    if coverage is not None:checks.record('DT5_data_order',True,coverage)
    checks.record('DT5_teacher_frozen',not any(p.grad is not None or p.requires_grad for p in teacher.parameters()))
    checks.record('finite_training',getattr(trainer,'nonfinite_steps',0)==0,dict(nonfinite_microbatches=getattr(trainer,'nonfinite_steps',0)))
    history=trainer.state.log_history
    trainer.optimizer.zero_grad(set_to_none=True);del trainer;gc.collect();torch.cuda.empty_cache()
    master=checks.attempt('master_final',lambda:score(student,ev['final']))
    round_to_saved_precision(student)
    # Preserve checkpoint even if a numerical scoring diagnostic fails.
    if rank==0:save_student(student,out/'final')
    state.wait_for_everyone()
    values={k:checks.attempt('post:'+k,lambda k=k:score(student,ev[k],teacher=teacher if k=='small' else None)) for k in ev}
    if dt4 and values['final'] is not None:
        checks.record('DT4',abs(values['final']['loss']-2.942554079)<=.002,dict(actual=values['final']['loss'],reference=2.942554079,tolerance=.002))
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
        result.update(flags=flags,unverified=bool(flags),log_history=history)
        write(out/'result.json',result)
        for label,value in [('parent',baseline),('pre-'+block,pre['final']),('post-'+block,values['final'])]:
            if value is not None:write(root/'perseq'/name/(label+'.json'),dict(value,run_id=name,flags=flags,unverified=bool(flags)))
    state.wait_for_everyone()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--dataset',required=True);p.add_argument('--arm',choices=('P300','P1T','X72'),required=True);p.add_argument('--block',choices=('W1','W9','S1'),required=True);p.add_argument('--dt4',action='store_true');a=p.parse_args();run(read(a.config),a.dataset,a.arm,a.block,a.dt4)
