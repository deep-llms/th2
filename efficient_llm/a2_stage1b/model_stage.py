"""Stage1b dataset dispatch with the Stage1 numerical implementation, using the unchanged A2 Trainer/KD."""
import argparse,json,os,time
from pathlib import Path
import torch
import numpy as np
from efficient_llm.a2.data import PackedRows,write_json
from efficient_llm.a2.model import load_parent,prune,bypass,KDModel
from efficient_llm.a2.measure import score,full_fp32
from efficient_llm.a2.train import heal,HealTrainer,RuntimeGuard,make_arguments,audit_order
from efficient_llm.a2.check import numerical_checks,parent_sanity
from efficient_llm.a2.protocol import lr_for_update,Run as OldRun
from efficient_llm.diagnostic_data import sha256
from .protocol import Run, REPLICATION_SPEC_SHA

def read(p):return json.loads(Path(p).read_text())

def evaluation(inputs,heldout=True):
 root=Path(inputs['original_inputs'])/'data';m=read(root/'manifest.json')
 result={n:PackedRows(root,n,verify=False) for n in ('final','tuning')}
 result['small']=PackedRows(root,'tuning',count=len(m['small_indices']),order=m['small_indices'],verify=False)
 if heldout:result['secondary']=PackedRows(inputs['heldout'],'tuning',verify=False)
 return result

def t8(student,teacher,rows,output,rank,world):
 from transformers import default_data_collator,set_seed
 set_seed(0);device=next(student.parameters()).device
 class Timer(RuntimeGuard):
  def __init__(self):super().__init__(300);self.times=[]
  def on_step_begin(self,*a,**kw):
   super().on_step_begin(*a,**kw);torch.cuda.synchronize();self.t=time.perf_counter()
  def on_step_end(self,*a,**kw):
   torch.cuda.synchronize();self.times.append(time.perf_counter()-self.t)
 timer=Timer();args=make_arguments(output,16,3e-4,4,world,cuda=True)
 trainer=HealTrainer(model=KDModel(student,256),teacher=teacher,args=args,train_dataset=rows,data_collator=default_data_collator,callbacks=[timer]);trainer.train()
 traces=[None]*world;torch.distributed.all_gather_object(traces,trainer.seen)
 coverage=audit_order(traces,rows.order.tolist(),world,4)
 if any(p.grad is not None or p.requires_grad for p in teacher.parameters()):raise ValueError('Teacher gradients')
 times=torch.tensor(timer.times,device=device);torch.distributed.all_reduce(times,op=torch.distributed.ReduceOp.MAX)
 if not np.allclose(timer.lrs,[lr_for_update(i,16,3e-4) for i in range(1,17)],rtol=1e-12,atol=0):raise ValueError('T8 LR mismatch')
 seconds=float(times[5:].mean());return dict(passed=True,step_seconds=seconds,tokens_per_second=262144/seconds,all_step_seconds=times.tolist(),coverage=coverage,microbatch=4,chunk=256,gpu=torch.cuda.get_device_name(),max_memory_bytes=torch.cuda.max_memory_allocated())

def main():
 p=argparse.ArgumentParser()
 p.add_argument('action',choices=('parent','numerical','throughput','t17','heal'))
 p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--parent',choices=('P300','P1T','S72'));p.add_argument('--run',type=Path);p.add_argument('--pre',type=Path);p.add_argument('--estimate',type=float,default=900)
 a=p.parse_args();inputs=read(a.inputs)
 if inputs.get('dataset')!='fwe' or sha256(inputs['replication_spec'])!=REPLICATION_SPEC_SHA or inputs['provenance'].get('stage1b_spec_sha256')!=REPLICATION_SPEC_SHA:raise ValueError('Stage1b identity changed')
 if a.output.exists():raise ValueError('Fresh output required')
 if sha256(inputs['spec'])!=inputs['provenance']['spec_sha256']:raise ValueError('Spec changed')
 from accelerate import PartialState
 state=PartialState();rank=state.process_index;world=state.num_processes
 if world!=8 or state.device.type!='cuda':raise ValueError('Eight B200 GPUs required')
 assets=read(Path(inputs['original_inputs'])/'assets.json');ev=evaluation(inputs)
 prov=inputs['provenance']
 if a.action=='parent' and a.parent=='S72' and not inputs['T3']['passed']:
  if rank==0:write_json(a.output,dict(status='complete',parent='S72',provenance=prov,T1={'passed':False,'reason':'Skipped because T3 failed'},sanity={'passed':False},baseline=None,pre=None))
  return
 if a.action=='parent':
  model=load_parent(assets,'fwe',a.parent,state.device);sanity=parent_sanity(model,ev['small'])
  amp=score(model,ev['small'])
  with full_fp32():fp32=score(model,ev['small'],amp=False)
  baseline={k:score(model,v) for k,v in ev.items()}
  with bypass(model,list(range(16,22))):pre={k:score(model,v) for k,v in ev.items()}
  del model;teacher=load_parent(assets,'fwe',a.parent,state.device,teacher=True);tv=score(teacher,ev['small'])
  passed=abs(amp['loss']-fp32['loss'])<.002
  result=dict(status='complete',parent=a.parent,provenance=prov,T1=dict(passed=passed,precision_gap=abs(amp['loss']-fp32['loss']),amp=amp,fp32=fp32,teacher_bf16=tv),sanity=sanity,baseline=baseline,pre=pre)
 elif a.action=='numerical':
  model=load_parent(assets,'fwe','P300',state.device)
  result=numerical_checks(model,ev['small'],{'S1':(8,9,10)},a.output.with_suffix('.work'))
  result['provenance']=prov
  if not result['passed']:raise ValueError('Stage1 numerical gate failed')
 elif a.action=='throughput':
  student=load_parent(assets,'fwe','P300',state.device);prune(student,range(16,22));teacher=load_parent(assets,'fwe','P300',state.device,teacher=True)
  result=t8(student,teacher,PackedRows(inputs['heal'],'heal',count=16*64,verify=False),a.output.with_suffix('.work'),rank,world);result['provenance']=prov
 else:
  if a.action=='t17':
   original=inputs['t17_reference'];run=OldRun('nem','P300','S1','0.05B',original['lr']);ds='nem'
   ev=evaluation(inputs,False)
   sec=read(Path(inputs['original_inputs'])/'secondary_map.json')['nem']
   if sec.get('path'):ev['secondary']=PackedRows(sec['path'],'final',verify=False)
   pre=dict(parent_final=original['val_parent'],teacher_final=original['val_teacher'],final={'loss':original['val_pre']},tuning={'loss':original['val_tune_pre']})
   if 'secondary' in ev:pre.update(parent_secondary=original['val_secondary_parent'],teacher_secondary=original['val_secondary_teacher'],secondary={'loss':original['val_secondary_pre']})
   rows=PackedRows(Path(inputs['original_inputs'])/'data','heal',count=run.steps*64,verify=False)
  else:
   raw=read(a.run);run=Run(raw['arm'],raw['budget'],raw['lr']);ds='fwe';pre=read(a.pre)
   if pre['provenance']!=prov:raise ValueError('Stale baseline')
   rows=PackedRows(inputs['heal'],'heal',count=run.steps*64,verify=False)
  student=load_parent(assets,ds,run.parent,state.device);prune(student,range(16,22));teacher=load_parent(assets,ds,run.teacher,state.device,teacher=True)
  run_prov=prov if a.action!='t17' else dict(prov,data_sha256=prov['original_data_sha256'],stage1_data_sha256=prov['data_sha256'],test='T17 original CulturaX stream')
  if a.action=='t17' and original['provenance']['data_sha256']!=run_prov['data_sha256']:raise ValueError('T17 original data differs')
  result=heal(student,teacher,rows,ev,a.output,run=run,microbatch=4,chunk=256,estimated_seconds=a.estimate,provenance=run_prov,pre_values=pre)
  if a.action=='t17':
   delta=result['val_post']-original['val_post'];check=dict(passed=abs(delta)<=.002,original_val=original['val_post'],actual_val=result['val_post'],difference=delta,original_run=original['run_id'],provenance=prov)
   if rank==0:write_json(a.output/'T17.json',check)
   if not check['passed']:raise ValueError('T17 differs by more than .002; stop')
  return
 if rank==0:write_json(a.output,result)
if __name__=='__main__':main()
