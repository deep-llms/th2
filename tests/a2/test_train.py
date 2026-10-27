import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from transformers import default_data_collator
from efficient_llm.a2.model import KDModel,prune
from efficient_llm.a2.train import HealTrainer,RuntimeGuard,make_arguments,audit_order,heal
from efficient_llm.a2.measure import score
from test_model import tiny_model

class Rows(torch.utils.data.Dataset):
    length=8
    def __init__(self,n):
        self.order=np.arange(n)
        self.ids=torch.randint(1,32,(n,8),generator=torch.Generator().manual_seed(9))
    def __len__(self):return len(self.order)
    def __getitem__(self,i):return dict(input_ids=self.ids[i],sample_id=int(self.order[i]))

class TrainerTests(unittest.TestCase):
    def test_accumulation_trainer_and_used_lr(self):
        teacher=tiny_model().requires_grad_(False)
        reference=copy.deepcopy(teacher).requires_grad_(True); prune(reference,[1,2])
        states=[]
        with tempfile.TemporaryDirectory() as d:
            for micro in (8,2):
                student=copy.deepcopy(reference)
                args=make_arguments(Path(d)/str(micro),12,1e-4,micro,1,cuda=False,global_sequences=8)
                guard=RuntimeGuard(60)
                trainer=HealTrainer(model=KDModel(student,3),teacher=teacher,args=args,
                    train_dataset=Rows(96),data_collator=default_data_collator,callbacks=[guard])
                trainer.train()
                self.assertEqual(guard.lrs[0],1e-5); self.assertEqual(guard.lrs[-1],0)
                self.assertTrue(audit_order([trainer.seen],range(96),1,micro)['exact_order'])
                states.append(copy.deepcopy(student.state_dict()))
                groups=trainer.optimizer.param_groups
                self.assertTrue(any(p is student.lm_head.weight for p in groups[0]['params']))
                self.assertTrue(all(p.ndim==1 for p in groups[1]['params']))
        for key in states[0]:torch.testing.assert_close(states[0][key],states[1][key],atol=2e-6,rtol=1e-5)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_complete_heal_export(self):
        class Fixture:
            steps=12; lr=1e-4
            def record(self): return dict(run_id='cpu-fixture',steps=12,lr=self.lr)
        teacher=tiny_model().requires_grad_(False)
        student=copy.deepcopy(teacher).requires_grad_(True);prune(student,[1,2])
        rows=Rows(3)
        pre=score(student,rows); parent=score(teacher,rows)['loss']
        with tempfile.TemporaryDirectory() as d:
            result=heal(student,teacher,Rows(96),dict(final=rows,tuning=rows,small=rows),Path(d)/'run',
                run=Fixture(),microbatch=2,chunk=3,estimated_seconds=60,provenance={'fixture':True},
                pre_values=dict(final=pre,tuning=pre,parent_final=parent,teacher_final=parent),global_sequences=8)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['roundtrip_difference'],0)
            self.assertEqual(result['coverage']['examples'],96)
            self.assertTrue((Path(d)/'run/final/model.safetensors').is_file())


    def test_nonfinite_runtime_estimates_refused(self):
        for value in (float('nan'),float('inf'),-1.,0.):
            with self.assertRaises(ValueError):RuntimeGuard(value)
