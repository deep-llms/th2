"""Fresh-optimizer Family healing using native Trainer and A2 forward KL."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import TrainerCallback,default_data_collator,set_seed
from efficient_llm.a2.train import HealTrainer,make_arguments,audit_order
from efficient_llm.a2.data import PackedRows
from efficient_llm.a2.model import autocast,round_to_saved_precision
from efficient_llm.a2.protocol import lr_for_update
from .common import read,write,sha,Checks,SPEC_SHA
from .model import KDModel,document_attention,prune,save_student,reload_student
from .evaluate import score_docs,load_model

class Rows(PackedRows):
    def __getitem__(self,index):
        value=super().__getitem__(index);value['document_starts']=self.boundaries(index);return value

class Trace(TrainerCallback):
    def __init__(self):self.lrs=[]
    def on_step_begin(self,args,state,control,optimizer=None,**kw):self.lrs.append(float(optimizer.param_groups[0]['lr']))

class FamilyTrainer(HealTrainer):
    def __init__(self,*args,trace_path,mask_documents=True,**kw):
        super().__init__(*args,**kw);self.trace_file=Path(trace_path).open('x',buffering=1);self.mask_documents=mask_documents;self.nonfinite_steps=0
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        ids=inputs['input_ids'];seen=inputs['sample_id'].detach().cpu().tolist();self.seen.append(seen)
        self.trace_file.write(json.dumps(dict(microbatch=len(self.seen)-1,optimizer_step=self.state.global_step,sample_ids=seen))+'\n')
        mask,pos=document_attention(inputs['document_starts']) if self.mask_documents else (None,None)
        with torch.no_grad(),autocast(ids.device):
            hidden=self.teacher.model(ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True).last_hidden_state
        outputs=model(input_ids=ids,teacher_hidden=hidden,teacher_weight=self.teacher.lm_head.weight,attention_mask=mask,position_ids=pos)
        if not torch.isfinite(outputs['loss'].detach()):
            self.nonfinite_steps+=1;print('FAMILY_NONFINITE_KD_FLAG',flush=True)
        return (outputs['loss'],outputs) if return_outputs else outputs['loss']
    def create_optimizer(self):
        if self.optimizer is None:
            # The tied input/output embedding belongs to the no-decay group.
            embedding=self.model.student.get_input_embeddings().weight
            decay=[];no_decay=[]
            for p in self.model.parameters():
                if p.requires_grad:(no_decay if p is embedding or p.ndim<2 else decay).append(p)
            self.optimizer=torch.optim.AdamW([dict(params=decay,weight_decay=.1),dict(params=no_decay,weight_decay=0.)],
                    lr=self.args.learning_rate,betas=(.9,.95),eps=1e-8)
        return self.optimizer


def run(cfg,item):
    from accelerate import PartialState
    state=PartialState();rank,world,device=state.process_index,state.num_processes,state.device
    root=Path(cfg['results_root']);out=root/'heals'/item['run_id']
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    state.wait_for_everyone();checks=Checks(out/f'tests-rank{rank}.json')
    checks.record('eight_ranks',world==8,world)
    entry=next(e for e in cfg['models']['checkpoints'] if e['label']==item['arm'])
    for f in entry['files']:checks.record('model_size:'+f['name'],(Path(entry['path'])/f['name']).stat().st_size==f['bytes'])
    set_seed(0);student,loading=load_model(entry,device);checks.record('complete_model_load',not any(loading.values()),loading)
    cache=read(root/'parents'/item['arm']/'result.json')
    baseline=cache['parent'];pre=cache['pre']
    keep=prune(student,list(range(24,33)))
    checks.record('FT1_keep_original_layers',keep==[i for i in range(36) if i not in range(24,33)],keep)
    teacher,teacher_loading=load_model(entry,device,teacher=True);teacher.eval().requires_grad_(False)
    checks.record('complete_teacher_load',not any(teacher_loading.values()),teacher_loading)
    for inherited in cache.get('flags',[]):checks.record('inherited_parent:'+inherited['test'],False,inherited)
    for inherited in cfg.get('input_flags',[]):
        if inherited.get('data') in (None,item['data']):checks.record('inherited_input',False,inherited)
    stream=cfg['mixture_stream'] if item.get('data')=='mix' else cfg['streams'][str(item['stream_seed'])]
    checks.record('stream_manifest',sha(Path(stream['path'])/'manifest.json')==stream['sha256'])
    rows=Rows(stream['path'],'heal',count=item['steps']*64,verify=False)
    mask_documents=cfg['mask_documents']
    checks.record('FT2_masking_enabled',mask_documents,dict(fallback='unmasked packing without resets' if not mask_documents else None))
    args=make_arguments(out/'trainer',item['steps'],item['lr'],1,world,cuda=True);trace=Trace()
    trainer=FamilyTrainer(model=KDModel(student,128),teacher=teacher,args=args,train_dataset=rows,
             data_collator=default_data_collator,callbacks=[trace],trace_path=out/f'sample-order-rank{rank}.jsonl',mask_documents=mask_documents)
    if rank==0:write(out/'recipe.json',dict(registered_run=item,training=args.to_dict(),kept=keep,
             parent_commit=entry['commit'],source_manifest=cfg['source_manifest'],stream=stream,
             optimizer='AdamW beta(.9,.95),eps1e-8,clip1;wd.1 except tied embeddings/norms; fresh',mask_documents=mask_documents))
    started=time.monotonic()
    try:trainer.train()
    finally:trainer.trace_file.close()
    train_seconds=time.monotonic()-started
    checks.record('FT4_updates',trainer.state.global_step==item['steps'],trainer.state.global_step)
    expected=[lr_for_update(i,item['steps'],item['lr']) for i in range(1,item['steps']+1)]
    checks.record('FT4_full_LR',len(trace.lrs)==len(expected) and np.allclose(trace.lrs,expected,rtol=1e-12,atol=0))
    traces=[None]*world;torch.distributed.all_gather_object(traces,trainer.seen)
    coverage=checks.attempt('FT4_data_order',lambda:audit_order(traces,rows.order.tolist(),world,1))
    if coverage is not None:checks.record('FT4_data_order',True,coverage)
    checks.record('finite_training',trainer.nonfinite_steps==0,trainer.nonfinite_steps)
    checks.record('teacher_frozen',not any(p.requires_grad or p.grad is not None for p in teacher.parameters()))
    trainer.optimizer.zero_grad(set_to_none=True);del trainer,teacher;gc.collect();torch.cuda.empty_cache()
    master=checks.attempt('master_final',lambda:score_docs(student,Path(cfg['cx_root'])/'final'))
    round_to_saved_precision(student)
    saved=True
    if rank==0:saved=checks.attempt('save_student',lambda:(save_student(student,out/'final'),True)[1]) is True
    available=torch.tensor(int(saved),device=device);torch.distributed.all_reduce(available,op=torch.distributed.ReduceOp.MIN)
    values={k:checks.attempt('post:'+k,lambda k=k:score_docs(student,Path(cfg['cx_root'])/k)) for k in ('final','tuning')}
    small=checks.attempt('roundtrip_before',lambda:score_docs(student,Path(cfg['cx_root'])/'tuning',limit=32))
    del student;gc.collect();torch.cuda.empty_cache()
    reloaded=checks.attempt('reload_model',lambda:reload_student(out/'final',device)) if available.item() else None
    ok=torch.tensor(int(reloaded is not None),device=device);torch.distributed.all_reduce(ok,op=torch.distributed.ReduceOp.MIN)
    score=checks.attempt('roundtrip_after',lambda:score_docs(reloaded,Path(cfg['cx_root'])/'tuning',limit=32)) if ok.item() else None
    difference=abs(score['loss']-small['loss']) if score and small else None
    checks.record('FT4_roundtrip',difference is not None and difference<=1e-6,dict(difference=difference))
    if values['tuning']:checks.record('tuning_divergence',values['tuning']['loss']<=pre['tuning']['loss'])
    rank_flags=[None]*world;torch.distributed.all_gather_object(rank_flags,[x for x in checks.items if not x['passed']])
    if rank==0:
        result=dict(run_id=item['run_id'],registered_run=item,status='complete',baseline=baseline,pre=pre,scores=values,
                    delta_post=values['final']['loss']-baseline['final']['loss'] if values['final'] else None,
                    master_final=master,lr_updates=trace.lrs,coverage=coverage,roundtrip_difference=difference,
                    training_seconds=train_seconds,elapsed_seconds=time.monotonic()-started,
                    flags=[f for flags in rank_flags for f in flags],raw_order_files=[dict(path=str(out/f'sample-order-rank{i}.jsonl'),sha256=sha(out/f'sample-order-rank{i}.jsonl')) for i in range(world)])
        write(out/'result.json',result)
        for name,value in [('parent',baseline['final']),('pre-BS1',pre['final']),('post-BS1',values['final'])]:
            if value is not None:write(root/'perseq'/item['run_id']/(name+'.json'),value)
    state.wait_for_everyone()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--item',required=True);a=p.parse_args();run(read(a.config),read(a.item))
