"""CPU: one then two ranks. Optional --cuda: one then eight B200 ranks.

Random native OpenSci fixture, NOT released weights or a scientific heal run.
GPU invocation requires external ownership/Accelerate/burn preflight.
"""
import os
import sys
cuda='--cuda' in sys.argv
if not cuda:os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.update(OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline')
import copy
import json
from pathlib import Path
import sys
import torch
from accelerate import PartialState
from transformers import default_data_collator
from test_model import tiny_model
from test_train import Rows
from efficient_llm.a2.model import KDModel,prune,autocast,forward_kl,round_to_saved_precision,save_student,reload_student
from efficient_llm.a2.train import HealTrainer,RuntimeGuard,make_arguments,audit_order
from efficient_llm.a2.data import write_json
from efficient_llm.a2.measure import score

state=PartialState(cpu=not cuda)
rank,world=state.process_index,state.num_processes
root=Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=True)
if cuda and (state.device.type!='cuda' or world not in (1,8) or 'B200' not in torch.cuda.get_device_name()):
    raise ValueError('CUDA smoke requires the authorized one/eight B200 setup')
teacher=tiny_model(**(dict(hidden_size=256,intermediate_size=512,num_attention_heads=4,
    num_key_value_heads=4,head_dim=64,vocab_size=1024) if cuda else {})).to(state.device).requires_grad_(False)
student=copy.deepcopy(teacher).requires_grad_(True);prune(student,[1,2])
head_check=None
if cuda:
    teacher.to(torch.bfloat16)
    torch.manual_seed(4)
    h=torch.randn(2,17,128,device=state.device,requires_grad=True)
    w=torch.randn(1024,128,device=state.device,requires_grad=True)
    th=torch.randn_like(h).bfloat16();tw=torch.randn_like(w).bfloat16()
    with autocast(state.device):
        reference=forward_kl(h,w,th,tw,5,recompute_head=False)
        recomputed=forward_kl(h,w,th,tw,5,recompute_head=True)
    gr=torch.autograd.grad(reference,(h,w));gc=torch.autograd.grad(recomputed,(h,w))
    torch.testing.assert_close(reference,recomputed,rtol=0,atol=0)
    for a,b in zip(gr,gc):torch.testing.assert_close(a,b,rtol=0,atol=0)
    head_check=dict(loss_difference=abs(reference.item()-recomputed.item()),
                    max_gradient_difference=max((a-b).abs().max().item() for a,b in zip(gr,gc)))
    del h,w,th,tw,gr,gc,reference,recomputed
global_sequences=64 if cuda else 8
micro=1 if cuda else 2//world
row_args=dict(length=64,vocab=1024) if cuda else {}
args=make_arguments(root/f'trainer-w{world}',12,1e-4,micro,world,cuda=cuda,global_sequences=global_sequences)
guard=RuntimeGuard(120)
trainer=HealTrainer(model=KDModel(student,3),teacher=teacher,args=args,
    train_dataset=Rows(12*global_sequences,**row_args),data_collator=default_data_collator,callbacks=[guard])
trainer.train()
traces=[None]*world
if world>1:torch.distributed.all_gather_object(traces,trainer.seen)
else:traces[0]=trainer.seen
coverage=audit_order(traces,range(12*global_sequences),world,micro)
# Uneven rows and fewer rows than ranks: no padding, duplicates or missing rows.
evaluation=[]
for count in (5,1):
    rows=Rows(count,**row_args)
    parallel=score(student,rows,teacher=teacher,chunk=3)
    serial=score(student,rows,teacher=teacher,chunk=3,distributed=False)
    assert parallel['tokens']==serial['tokens']==count*(rows.length-1)
    gap=max(abs(a-b) for a,b in zip(parallel['per_sequence_nll'],serial['per_sequence_nll']))
    assert gap<(2e-4 if cuda else 2e-6)
    assert abs(parallel['final_kl']-serial['final_kl'])<(2e-4 if cuda else 1e-6)
    evaluation.append(dict(sequences=count,tokens=parallel['tokens'],max_nll_difference=gap))
if rank==0:
    if world==1:
        torch.save(student.state_dict(),root/'reference.pt')
        write_json(root/'reference_lrs.json',guard.lrs)
        write_json(root/'reference.json',dict(status='passed',coverage=coverage,evaluation=evaluation,head_check=head_check))
    else:
        ref=torch.load(root/'reference.pt',weights_only=True)
        max_difference=0.
        for key,value in student.state_dict().items():
            torch.testing.assert_close(value.cpu(),ref[key].cpu(),atol=2e-5 if cuda else 2e-6,rtol=1e-4 if cuda else 1e-5)
            max_difference=max(max_difference,(value-ref[key]).abs().max().item())
        assert guard.lrs==json.loads((root/'reference_lrs.json').read_text())
        write_json(root/'ddp_result.json',dict(status='passed',coverage=coverage,cuda=cuda,head_check=head_check,
            max_parameter_difference=max_difference,lr_updates=guard.lrs,evaluation=evaluation,
            teacher_no_gradient=all(p.grad is None for p in teacher.parameters())))
state.wait_for_everyone()
if cuda:
    # Real BF16 export/reload under the same AMP scoring path, with uneven sharding.
    round_to_saved_precision(student)
    rows=Rows(9,**row_args)
    before=score(student,rows)
    path=root/f'roundtrip-w{world}'
    if rank==0:save_student(student,path)
    state.wait_for_everyone()
    restored=reload_student(path,state.device)
    after=score(restored,rows)
    difference=abs(before['loss']-after['loss'])
    assert difference<=1e-6
    if rank==0:write_json(root/f'roundtrip-w{world}.json',dict(status='passed',difference=difference))
    state.wait_for_everyone()
if world>1:torch.distributed.destroy_process_group()
