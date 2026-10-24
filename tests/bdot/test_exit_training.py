import copy
from dataclasses import replace
import tempfile
import unittest

import torch
from torch.nn import functional as F

from test_model import batch
from test_stopping import stopped
from bdot.model import BDOTForCausalLM
from bdot.exit_training import install, loss_sums


def dual(partial=True, kl=.5):
    m = stopped(partial, 0.)
    m.config.bdot_stop = dict(train_random_prob=0., min_steps=2, exit_steps=[2])
    m.stop_controller.options = replace(m.stop_controller.options, exit_steps=(2,))
    m.config.bdot_exit_training = dict(depth=2, ce_weight=.5, kl_weight=kl, chunk_size=3)
    install(m)
    return m


class ExitTrainingTests(unittest.TestCase):
    def test_chunked_loss_matches_direct_formula_and_teacher_is_detached(self):
        torch.manual_seed(91)
        a = torch.randn(2, 5, 37, requires_grad=True)
        t = torch.randn_like(a, requires_grad=True)
        labels = torch.tensor([[-100,2,3,-100,4],[-100,-100,-100,-100,-100]])
        sums = loss_sums(a, t, labels, chunk_size=3)
        logp = a[:,:-1].log_softmax(-1); logq = t[:,:-1].detach().log_softmax(-1)
        ce = F.cross_entropy(a[:,:-1].reshape(-1,37), labels[:,1:].reshape(-1),
                             ignore_index=-100, reduction='sum')
        kl = (logq.exp()*(logq-logp)).sum(-1).masked_fill(labels[:,1:]==-100,0).sum()
        torch.testing.assert_close(sums, torch.stack((ce,kl)), atol=2e-6, rtol=1e-6)
        expected = torch.autograd.grad(ce+kl,a,retain_graph=True)[0]
        sums.sum().backward()
        torch.testing.assert_close(a.grad,expected,atol=1e-6,rtol=1e-5)
        self.assertIsNone(t.grad)
        empty = torch.full_like(labels,-100)
        self.assertEqual(float(loss_sums(a,t,empty).sum().detach()),0.)

    def test_dual_matches_independent_two_pass_objective_and_full_stats(self):
        for partial in (False,True):
            for kl in (0.,.5):
                a = dual(partial,kl); b = copy.deepcopy(a)
                del b.exit_options; del b.config.bdot_exit_training
                inputs=batch(); n=inputs['labels'][:,1:].ne(-100).sum()
                full = b(**inputs); full_counts=b.load_counts.clone(); full_metrics=b.metric_totals.clone()
                early = b(**dict(inputs,labels=None),bdot_stop_depths=torch.full_like(inputs['labels'],2))
                ls=loss_sums(early.logits,full.logits,inputs['labels'],3,kl>0)
                reference=full.loss+(.5*ls[0]+kl*ls[1])/n
                actual=a(**inputs)
                torch.testing.assert_close(actual.loss,reference,atol=1e-6,rtol=1e-6)
                torch.testing.assert_close(actual.logits,full.logits,atol=0,rtol=0)
                actual.loss.backward(); reference.backward()
                for name,p in a.named_parameters():
                    q=b.get_parameter(name)
                    if p.grad is None:self.assertIsNone(q.grad,name)
                    else:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4,msg=name)
                torch.testing.assert_close(a.load_counts,full_counts,atol=0,rtol=0)
                torch.testing.assert_close(a.metric_totals,full_metrics,atol=0,rtol=0)
                self.assertEqual(int(a.routing_counter),1)
                self.assertTrue(a.stop_controller.last_depths.eq(4).all())

    def test_accumulation_and_empty_local_targets(self):
        for kl in (0.,.5):
            a=dual(kl=kl);b=copy.deepcopy(a)
            inputs=batch();inputs['labels'][0].fill_(-100)
            count=inputs['labels'][:,1:].ne(-100).sum();decisions=inputs['input_ids'].numel()*4
            full=a(**inputs,num_items_in_batch=count,bdot_num_decisions=decisions).loss
            full.backward();summed=0
            for row in range(2):
                loss=b(**{k:v[row:row+1] for k,v in inputs.items()},
                       num_items_in_batch=count,bdot_num_decisions=decisions).loss
                loss.backward();summed=summed+loss.detach()
            torch.testing.assert_close(full.detach(),summed,atol=1e-6,rtol=1e-6)
            for p,q in zip(a.parameters(),b.parameters()):
                if p.grad is None:self.assertIsNone(q.grad)
                else:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)
            a.finish_update();b.finish_update()
            torch.testing.assert_close(a.exit_training_totals,b.exit_training_totals,atol=2e-6,rtol=1e-6)

    def test_checkpoint_replay_and_serialization(self):
        a=dual();b=copy.deepcopy(a)
        b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        x=a(**batch());y=b(**batch());x.loss.backward();y.loss.backward()
        torch.testing.assert_close(x.loss,y.loss,atol=0,rtol=0)
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is None:self.assertIsNone(q.grad)
            else:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
        a.finish_update()
        with tempfile.TemporaryDirectory() as tmp:
            a.save_pretrained(tmp)
            b=BDOTForCausalLM.from_pretrained(tmp,attn_implementation='sdpa').train()
        self.assertFalse(b.exit_metric_totals.any())
        self.assertFalse(any(p.requires_grad for p in b.stop_controller.parameters()))
        for name,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[name],atol=0,rtol=0)
        torch.testing.assert_close(a(**batch()).loss,b(**batch()).loss,atol=0,rtol=0)

    def test_eval_bypasses_auxiliary_and_single_exit_constraints(self):
        m=dual().eval();m.stop_controller.policy='learned'
        with torch.no_grad():m.stop_controller.bias.fill_(20)
        m(**batch());self.assertTrue(m.stop_controller.last_depths.eq(2).all())
        m.stop_controller.policy='fixed';m.stop_controller.fixed_depth=3
        with self.assertRaises(ValueError):m(**batch())
        m.stop_controller.policy='none'
        with self.assertRaises(ValueError):m(**batch(),bdot_stop_depths=torch.full((2,6),3))
        self.assertFalse(m.exit_training_totals.any())
        a=dual().train()
        with self.assertRaises(ValueError):a(**batch(),logits_to_keep=1)

    def test_late_drop_has_only_one_opportunity_and_resumes_rng(self):
        m=stopped(True,.5)
        m.stop_controller.options=replace(m.stop_controller.options,exit_steps=(2,))
        x=batch();m(**x)
        self.assertTrue(torch.isin(m.stop_controller.last_depths,torch.tensor([2,4])).all())
        m.routing_counter.zero_();m(**x);a=m.stop_controller.last_depths.clone()
        m.routing_counter.zero_();m(**x)
        self.assertTrue(torch.equal(a,m.stop_controller.last_depths))

    def test_policy_selection_accounts_for_degraded_full_path(self):
        from eval.stop_v2_study import select_policy
        full=dict(policy=dict(policy='none'),loss=3.02,mean_middle_steps=8.)
        early=dict(policy=dict(policy='learned',threshold=.5),loss=3.025,mean_middle_steps=6.8)
        chosen,proof=select_policy([full,early],3.)
        self.assertEqual(chosen,full);self.assertFalse(proof['feasible'])
        full['loss']=3.;early['loss']=3.005
        chosen,proof=select_policy([full,early],3.)
        self.assertEqual(chosen,early);self.assertTrue(proof['feasible'])


if __name__=='__main__':unittest.main()
