"""Teacher-free, shifted next-token CE; Stage1 optimizer/scheduler/order."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Trainer
from efficient_llm.a2.train import HealTrainer

def ce_chunk(hidden,weight,labels):
    return F.cross_entropy(F.linear(hidden,weight).float().flatten(0,1),labels.reshape(-1),reduction='sum')

class LMModel(nn.Module):
    def __init__(self,student,chunk=256):
        super().__init__();self.student=student;self.config=student.config;self.chunk=chunk
    def forward(self,input_ids):
        hidden=self.student.model(input_ids,use_cache=False,return_dict=True).last_hidden_state
        n=input_ids.shape[1]-1
        if n<1:raise ValueError('Need next-token targets')
        loss=hidden.new_zeros((),dtype=torch.float32)
        for start in range(0,n,self.chunk):
            end=min(start+self.chunk,n)
            args=(hidden[:,start:end],self.student.lm_head.weight,input_ids[:,start+1:end+1])
            loss+=checkpoint(ce_chunk,*args,use_reentrant=False,preserve_rng_state=False) if torch.is_grad_enabled() else ce_chunk(*args)
        return {'loss':loss/(input_ids.shape[0]*n)}

class LMTrainer(Trainer):
    # Reuse the exact existing Stage1 recipe without initializing HealTrainer,
    # whose constructor intentionally requires a teacher.
    _get_train_sampler=HealTrainer._get_train_sampler
    create_optimizer=HealTrainer.create_optimizer
    create_scheduler=HealTrainer.create_scheduler
    def __init__(self,*args,trace_path,**kw):
        super().__init__(*args,**kw);self.model_accepts_loss_kwargs=False;self.seen=[]
        if self.args.gradient_checkpointing or self.args.label_smoothing_factor:raise ValueError('Changed LM recipe')
        self.trace_file=Path(trace_path).open('x',buffering=1)
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        seen=inputs['sample_id'].detach().cpu().tolist();self.seen.append(seen)
        self.trace_file.write(json.dumps(dict(microbatch=len(self.seen)-1,optimizer_step=self.state.global_step,sample_ids=seen))+'\n')
        outputs=model(input_ids=inputs['input_ids'])
        if not torch.isfinite(outputs['loss'].detach()):
            self.nonfinite_steps=getattr(self,'nonfinite_steps',0)+1
            print('F11_NONFINITE_CE_FLAG',flush=True)
        return (outputs['loss'],outputs) if return_outputs else outputs['loss']
