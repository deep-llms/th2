"""Two-rank CPU/Gloo fixture; never accesses GPU devices or project datasets."""
import argparse
import copy
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers import TrainerCallback, default_data_collator, set_seed
from tests.a2.test_model import tiny_model
from efficient_llm.a2.model import KDModel, prune
from efficient_llm.a2.train import make_arguments, audit_order
from efficient_llm.a2_depth.heal_short import ReportTrainer, Trace
from efficient_llm.a2_depth.resume import enable_checkpoints, CheckpointAudit, select_checkpoint, restore_audit
from efficient_llm.a2_depth.common import write


class Probe(TrainerCallback):
    def __init__(self,stop=None):self.stop=stop;self.values=[]
    def on_step_end(self,args,state,control,**kw):
        self.values.append([random.random(),float(np.random.random()),float(torch.rand(()))])
        if state.global_step==self.stop:control.should_training_stop=True;control.should_save=True


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    state=PartialState(cpu=True);assert state.num_processes==2 and state.device.type=='cpu'
    # The test launcher sets ACCELERATE_TORCH_DEVICE=cpu so each fresh
    # TrainingArguments/PartialState reset retains an unindexed CPU device.
    teacher=tiny_model();initial=copy.deepcopy(teacher);prune(initial,[1])
    rows=[dict(input_ids=torch.tensor([1,2+i%20,3,4,5]),sample_id=i) for i in range(64)]
    identity=dict(steps=16,global_sequences=4,microbatch=1,lr=.001,data='fixture',teacher='fixture',set='fixture')
    def run(path,stop=None,resume=None):
        set_seed(0);trace=Trace();probe=Probe(stop)
        args=enable_checkpoints(make_arguments(path,16,.001,1,2,cuda=False,global_sequences=4),every=4)
        trainer=ReportTrainer(model=KDModel(copy.deepcopy(initial),2),teacher=copy.deepcopy(teacher),args=args,
            train_dataset=rows,data_collator=default_data_collator,callbacks=[trace,probe])
        trainer.add_callback(CheckpointAudit(trainer,trace,identity))
        if resume:restore_audit(trainer,trace,resume)
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        return trainer,trace,probe
    full,ft,fp=run(a.output/'full')
    first,st,sp=run(a.output/'resume',stop=4)
    checkpoint,rejections=select_checkpoint(a.output/'resume',identity,2)
    assert checkpoint and not rejections
    resumed,rt,rp=run(a.output/'resume',resume=checkpoint)
    assert full.seen==resumed.seen and ft.lrs==rt.lrs and fp.values==sp.values+rp.values
    for key,value in full.model.state_dict().items():
        torch.testing.assert_close(value,resumed.model.state_dict()[key],rtol=0,atol=0)
    full_opt,resumed_opt=full.optimizer.state_dict(),resumed.optimizer.state_dict()
    assert full_opt['param_groups']==resumed_opt['param_groups']
    for key,values in full_opt['state'].items():
        for name,value in values.items():torch.testing.assert_close(value,resumed_opt['state'][key][name],rtol=0,atol=0)
    assert full.lr_scheduler.state_dict()==resumed.lr_scheduler.state_dict()
    traces=[None]*2;dist.all_gather_object(traces,resumed.seen)
    order=audit_order(traces,list(range(64)),2,1)
    write(a.output/f'verified-rank{state.process_index}.json',dict(passed=True,world=2,device='cpu',
        global_step=resumed.state.global_step,exact_weights_optimizer_scheduler_RNG_LR=True,order=order))
    state.wait_for_everyone();dist.destroy_process_group()


if __name__=='__main__':main()
