"""T11: actual BDOTTrainer/DDP accumulation versus a full-batch reference.
Run with torchrun --standalone --nproc_per_node=2 tests/bdot/ddp_probe.py OUTPUT.
Use --device cuda with an explicitly allocated CUDA_VISIBLE_DEVICES for NCCL.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
import copy
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from datasets import Dataset
from transformers import TrainerCallback
from bdot.trainer import BDOTTrainer
from train import BaselineTrainingArguments
from packed_training import PackedCollator, model_inputs
from test_model import model

torch.set_num_threads(1)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output');parser.add_argument('--checkpointing',action='store_true')
parser.add_argument('--soft',action='store_true')
parser.add_argument('--no-reuse',action='store_true')
parser.add_argument('--home-only',action='store_true')
parser.add_argument('--expert-backend',choices=['reference','grouped'],default='reference')
parser.add_argument('--bf16',action='store_true')
parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
settings=parser.parse_args()
if settings.device=='cuda':
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('Allocate CUDA_VISIBLE_DEVICES explicitly')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
device=torch.device('cuda',int(os.environ['LOCAL_RANK'])) if settings.device=='cuda' else torch.device('cpu')
root=Path(settings.output);root.mkdir(parents=True,exist_ok=True)
rank=int(os.environ['RANK'])
rows=[dict(input_ids=[i+2,3,4,5,6,7],labels=[i+2,3,4,5,6,7],attention_mask=[1]*6,
           segments=[0,0,1,1,1,1] if i%2==0 else [0]*6) for i in range(4)]
ds=Dataset.from_list(rows)

class ForcedTrainer(BDOTTrainer):
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        if settings.no_reuse or settings.home_only:
            path = [1,0,2,3] if rank == 1 and not settings.home_only else [0,1,2,3]
            forced = torch.tensor(path, device=device)[:,None,None].expand(4,*inputs['input_ids'].shape)
        else:
            forced = torch.full((4,*inputs['input_ids'].shape),rank,dtype=torch.long,device=device)
        inputs=dict(inputs,bdot_forced_routes=forced)
        return super().compute_loss(model,inputs,return_outputs,num_items_in_batch)

class Gradients(TrainerCallback):
    def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
        self.grads={n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.named_parameters()}
        self.counts=model.load_counts.clone()

m=model(r=1 if settings.no_reuse or settings.home_only else 3,
        reuse=not settings.no_reuse,
        curriculum='A' if settings.home_only else 'B' if settings.soft else 'none',default_prior=0.,
        expert_backend=settings.expert_backend).to(device)
ref=copy.deepcopy(m)
args=BaselineTrainingArguments(output_dir=str(root/'training'),use_cpu=settings.device=='cpu',report_to='none',bf16=settings.bf16,
    max_steps=1,save_strategy='no',logging_strategy='no',per_device_train_batch_size=1,
    gradient_accumulation_steps=2,learning_rate=.001,weight_decay=.1,max_grad_norm=0.,
    remove_unused_columns=False,disable_tqdm=True,dataloader_pin_memory=False,
    train_sampling_strategy='sequential',seed=7,data_seed=0,
    gradient_checkpointing=settings.checkpointing)
probe=Gradients()
trainer_class=BDOTTrainer if settings.soft else ForcedTrainer
t=trainer_class(model=m,args=args,train_dataset=ds,data_collator=PackedCollator(),callbacks=[probe])
t.create_optimizer()
name_by_id={id(p):n for n,p in m.named_parameters()}
refs=dict(ref.named_parameters())
groups=[{**{k:v for k,v in group.items() if k!='params'},
         'params':[refs[name_by_id[id(p)]] for p in group['params']]} for group in t.optimizer.param_groups]
opt=torch.optim.AdamW(groups,lr=.001)
if settings.bf16:
    # BF16 batch-shape rounding can flip almost-zero gradients; a fresh AdamW
    # first step converts those into a full +/-LR update even in the reference
    # backend. Use identical nonzero moments to test distributed optimizer math
    # without that sign discontinuity, also checking unused momentum is skipped.
    for optimizer in (t.optimizer,opt):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                step_device=parameter.device if group.get('fused') or group.get('capturable') else 'cpu'
                optimizer.state[parameter] = dict(step=torch.tensor(5.,device=step_device),
                    exp_avg=torch.full_like(parameter,.001),
                    exp_avg_sq=torch.full_like(parameter,.0001))
inputs=model_inputs({k:v.to(device) for k,v in PackedCollator()(rows).items()})
forced=torch.tensor([0,1,0,1],device=device)[None,:,None].expand(4,4,6)
if settings.no_reuse or settings.home_only:
    paths = [[1,0,2,3] if i%2 and not settings.home_only else [0,1,2,3] for i in range(4)]
    forced = torch.tensor(paths, device=device).T[:,:,None].expand(4,4,6)
with torch.autocast(device.type,dtype=torch.bfloat16,enabled=settings.bf16):
    loss=ref(**inputs,**({} if settings.soft else dict(bdot_forced_routes=forced))).loss
loss.backward()
t.train()
max_error=0.
for name,p in ref.named_parameters():
    a,b=probe.grads[name],p.grad
    if b is None:
        assert a is None,(name,'globally unused got a gradient')
    else:
        assert a is not None,name
        torch.testing.assert_close(a,b,atol=1e-3 if settings.bf16 else 1e-5,
                                   rtol=.03 if settings.bf16 else 1e-5)
        max_error=max(max_error,float((a-b).abs().max()))
counts=probe.counts.clone();torch.distributed.all_reduce(counts)
torch.testing.assert_close(counts,ref.load_counts,atol=0,rtol=0)
shares=ref.load_counts.float()/ref.load_counts.sum(-1,keepdim=True)
expected=ref.options.balance_rate*(torch.full((4,),.25,device=device)-shares.mean(0)).sign()
torch.testing.assert_close(m.balance_bias,expected,atol=0,rtol=0)
expected_lm = float(loss.detach()) - ref.options.z_loss_coef * float(ref.metric_totals[2]/ref.metric_totals[3])
assert abs(m.last_update_metrics['lm_loss_last_update']-expected_lm) < 1e-5
opt.step()
for name,p in m.named_parameters():
    torch.testing.assert_close(p,refs[name],atol=1e-4 if settings.bf16 else 1e-5,
                               rtol=.01 if settings.bf16 else 1e-5,msg=name)
(root/f'rank{rank}.json').write_text(json.dumps(dict(status='passed',max_gradient_error=max_error,
    counts=counts.tolist(),balance_bias=m.balance_bias.tolist(),reference_loss=loss.item(),
    checkpointing=settings.checkpointing,soft=settings.soft,device=str(device),torch=torch.__version__,
    expert_backend=settings.expert_backend,bf16=settings.bf16,no_reuse=settings.no_reuse,
    home_only=settings.home_only))+'\n')
torch.distributed.barrier();torch.distributed.destroy_process_group()
