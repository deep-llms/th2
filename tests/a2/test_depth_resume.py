import copy
import random
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from transformers import TrainerCallback, default_data_collator, set_seed
from tests.a2.test_model import tiny_model
from efficient_llm.a2.model import KDModel, prune
from efficient_llm.a2.train import make_arguments
from efficient_llm.a2_depth.heal_short import ReportTrainer, Trace
from efficient_llm.a2_depth.resume import enable_checkpoints, CheckpointAudit, select_checkpoint, restore_audit
from efficient_llm.a2_depth.common import read
from efficient_llm.a2.protocol import warmup_steps, lr_for_update


class RandomProbe(TrainerCallback):
    def __init__(self, stop=None): self.values=[];self.stop=stop
    def on_step_end(self,args,state,control,**kwargs):
        self.values.append((random.random(),float(np.random.random()),float(torch.rand(()))))
        if state.global_step==self.stop:control.should_training_stop=True;control.should_save=True


class DepthResumeTests(unittest.TestCase):
    def test_native_full_state_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);teacher=tiny_model();initial=copy.deepcopy(teacher);prune(initial,[1])
            rows=[dict(input_ids=torch.tensor([1,2+i%20,3,4,5]),sample_id=i) for i in range(32)]
            identity=dict(steps=16,global_sequences=2,microbatch=1,lr=.001,data='fixture',teacher='fixture',set='fixture')
            def run(path,stop=None,resume=None):
                set_seed(0)
                model=KDModel(copy.deepcopy(initial),2)
                args=enable_checkpoints(make_arguments(path,16,.001,1,1,cuda=False,global_sequences=2),every=4)
                trace=Trace();probe=RandomProbe(stop)
                trainer=ReportTrainer(model=model,teacher=copy.deepcopy(teacher),args=args,train_dataset=rows,
                    data_collator=default_data_collator,callbacks=[trace,probe])
                trainer.add_callback(CheckpointAudit(trainer,trace,identity))
                if resume:restore_audit(trainer,trace,resume)
                trainer.train(resume_from_checkpoint=str(resume) if resume else None)
                return trainer,trace,probe
            full,ft,fp=run(root/'full')
            first,st,sp=run(root/'resume',stop=4)
            checkpoint,rejected=select_checkpoint(root/'resume',identity,1)
            self.assertFalse(rejected);self.assertEqual(checkpoint.name,'checkpoint-4')
            resumed,rt,rp=run(root/'resume',resume=checkpoint)
            self.assertEqual(full.seen,resumed.seen)
            self.assertEqual(ft.lrs,rt.lrs)
            self.assertEqual(fp.values,sp.values+rp.values)
            for key,value in full.model.state_dict().items():
                torch.testing.assert_close(value,resumed.model.state_dict()[key],rtol=0,atol=0)
            a,b=full.optimizer.state_dict(),resumed.optimizer.state_dict()
            self.assertEqual(a['param_groups'],b['param_groups'])
            for key,state in a['state'].items():
                for name,value in state.items():torch.testing.assert_close(value,b['state'][key][name],rtol=0,atol=0)
            self.assertEqual(full.lr_scheduler.state_dict(),resumed.lr_scheduler.state_dict())
            final,rejected=select_checkpoint(root/'resume',identity,1)
            self.assertEqual(final.name,'checkpoint-16');self.assertFalse(rejected)
            (final/'optimizer.pt').write_bytes(b'broken')
            previous,rejected=select_checkpoint(root/'resume',identity,1)
            self.assertEqual(previous.name,'checkpoint-12');self.assertEqual(len(rejected),1)
            self.assertIsNone(select_checkpoint(root/'resume',dict(identity,steps=17),1)[0])
            self.assertEqual(read(previous/'trainer_state.json')['global_step'],12)

    def test_registered_C_schedule_and_prefix_counts(self):
        self.assertEqual(warmup_steps(48828),977)
        self.assertEqual(48828*64,3124992)
        self.assertEqual(12207*64,781248)
        self.assertEqual(lr_for_update(977,48828,.001),.001)
        self.assertEqual(lr_for_update(48828,48828,.001),0)


if __name__=='__main__':unittest.main()
