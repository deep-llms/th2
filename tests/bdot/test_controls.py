import copy
import unittest
import torch
from test_model import model,batch


class ControlTests(unittest.TestCase):
    def test_random_control_and_overrides(self):
        m=model(routing='random',curriculum='none',r=3);m.eval();m.capture_routes=True
        a=m(**batch());first=m.last_routes.clone();m(**batch());second=m.last_routes.clone()
        self.assertFalse(torch.equal(first,second))
        m.eval_counter=0;b=m(**batch())
        torch.testing.assert_close(a.logits,b.logits,atol=0,rtol=0)
        m.routing_override='fixed';m(**batch())
        self.assertTrue((m.last_routes==torch.arange(4)[:,None,None]).all())

    def test_soft_ties_single_candidate_and_finite_gradients(self):
        m=model(reuse=False,curriculum='B');m.capture_routes=True
        with torch.no_grad():
            for layer in m.routed_steps:
                layer.router.weight.zero_();layer.router.bias.zero_()
        out=m(**batch());out.loss.backward()
        self.assertTrue(torch.isfinite(out.loss))
        self.assertTrue((m.last_routes.sort(0).values==torch.arange(4)[:,None,None]).all())
        for p in m.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())

    def test_validation_has_no_auxiliary_loss(self):
        a=model(z_loss_coef=1.);b=copy.deepcopy(a)
        a.eval();b.eval()
        a.options=copy.copy(a.options)  # output loss in eval must equal native CE
        out=a(**batch())
        inputs=batch();labels=inputs.pop('labels')
        logits=b(**inputs).logits
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,37),labels[:,1:].reshape(-1),ignore_index=-100)
        torch.testing.assert_close(out.loss,loss)

    def test_reuse_metric_counts_tokens_not_repeated_decisions(self):
        from types import SimpleNamespace
        from datasets import Dataset
        from packed_evaluation import evaluate_packed
        from packed_training import PackedCollator
        m=model(r=3,curriculum='none');m.capture_routes=True
        # Every token follows 0,0,2,3: one repeat in four decisions, but ALL
        # tokens reuse a module. Replacing the second choice by 1 removes reuse.
        with torch.no_grad():
            for step,layer in enumerate(m.routed_steps):
                layer.router.weight.zero_();layer.router.visits.zero_()
                layer.router.bias.zero_();layer.router.bias[0 if step==1 else step]=5.
        ds=Dataset.from_list([dict(input_ids=[1,2,3,4],labels=[1,2,3,4],
                                   attention_mask=[1]*4,segments=[0]*4)])
        from packed_training import model_inputs
        m(**model_inputs(PackedCollator()([ds[0]]))).loss.backward()
        m.finish_update()
        stats=m.last_update_metrics
        self.assertEqual(stats['reuse_rate_last_update'],1.)
        self.assertEqual(stats['off_default_step_1_last_update'],1.)
        self.assertEqual(stats['module_0_decisions_last_update'],8)
        self.assertEqual(stats['module_1_decisions_last_update'],0)
        self.assertIsNone(stats['module_load_max_min_ratio_last_update'])
        self.assertGreater(stats['router_entropy_last_update'],0.)
        from contextlib import nullcontext
        accelerator=SimpleNamespace(device=torch.device('cpu'),process_index=0,num_processes=1,
                                    reduce=lambda x,**kwargs:x,autocast=nullcontext)
        metrics=evaluate_packed(m,ds,PackedCollator(),accelerator)
        self.assertEqual(metrics['reuse_rate'],1.)
        self.assertEqual(metrics['off_default_share'],.25)
        m.routing_override='fixed'
        metrics=evaluate_packed(m,ds,PackedCollator(),accelerator)
        self.assertEqual(metrics['reuse_rate'],0.)
        self.assertEqual(metrics['off_default_share'],0.)

if __name__=='__main__':unittest.main()
