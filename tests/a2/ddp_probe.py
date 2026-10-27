"""Run once normally, then with torchrun --nproc_per_node=2; compare updates/order."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline')
import copy
import json
from pathlib import Path
import sys
import torch
from accelerate import PartialState
from transformers import default_data_collator
from test_model import tiny_model
from test_train import Rows
from efficient_llm.a2.model import KDModel,prune
from efficient_llm.a2.train import HealTrainer,RuntimeGuard,make_arguments,audit_order
from efficient_llm.a2.data import write_json
from efficient_llm.a2.measure import score

state=PartialState(cpu=True)
rank,world=state.process_index,state.num_processes
root=Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=True)
teacher=tiny_model().requires_grad_(False)
student=copy.deepcopy(teacher).requires_grad_(True);prune(student,[1,2])
args=make_arguments(root/f'trainer-w{world}',12,1e-4,2//world,world,cuda=False,global_sequences=8)
guard=RuntimeGuard(120)
trainer=HealTrainer(model=KDModel(student,3),teacher=teacher,args=args,
    train_dataset=Rows(96),data_collator=default_data_collator,callbacks=[guard])
trainer.train()
traces=[None]*world
if world>1:torch.distributed.all_gather_object(traces,trainer.seen)
else:traces[0]=trainer.seen
coverage=audit_order(traces,range(96),world,2//world)
# Uneven rows and fewer rows than ranks: no padding, duplicates or missing rows.
evaluation=[]
for count in (5,1):
    rows=Rows(count)
    parallel=score(student,rows,teacher=teacher,chunk=3)
    serial=score(student,rows,teacher=teacher,chunk=3,distributed=False)
    assert parallel['tokens']==serial['tokens']==count*7
    gap=max(abs(a-b) for a,b in zip(parallel['per_sequence_nll'],serial['per_sequence_nll']))
    assert gap<2e-6
    assert abs(parallel['final_kl']-serial['final_kl'])<1e-6
    evaluation.append(dict(sequences=count,tokens=parallel['tokens'],max_nll_difference=gap))
if rank==0:
    if world==1:
        torch.save(student.state_dict(),root/'reference.pt')
        write_json(root/'reference_lrs.json',guard.lrs)
    else:
        ref=torch.load(root/'reference.pt',weights_only=True)
        max_difference=0.
        for key,value in student.state_dict().items():
            torch.testing.assert_close(value,ref[key],atol=2e-6,rtol=1e-5)
            max_difference=max(max_difference,(value-ref[key]).abs().max().item())
        assert guard.lrs==json.loads((root/'reference_lrs.json').read_text())
        write_json(root/'ddp_result.json',dict(status='passed',coverage=coverage,
            max_parameter_difference=max_difference,lr_updates=guard.lrs,evaluation=evaluation,
            teacher_no_gradient=all(p.grad is None for p in teacher.parameters())))
state.wait_for_everyone()
if world>1:torch.distributed.destroy_process_group()
