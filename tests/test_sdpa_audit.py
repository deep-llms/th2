"""The profiler must preserve gradients and attribute the complete objective."""
import copy
from contextlib import nullcontext
import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers import TrainingArguments
from deep_kv import ARMS
from deep_kv.model import DeepKV
from deep_kv.training import DeepKVTrainer
from deep_kv.sdpa_audit import SDPAAudit
from tests.test_deep_kv import config
from scripts.benchmark_document_training import Collator


class SDPAAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_arms_objective_gradients_and_call_attribution(self):
        rows=[dict(input_ids=[3,4,31,8,9,10,31],segments=[0,0,0,1,1,1,1])]
        batch=Collator('sdpa_isolated')(rows)
        for arm in ARMS:
            for checkpoint in (False, True):
                with self.subTest(arm=arm, checkpoint=checkpoint), tempfile.TemporaryDirectory() as folder:
                    model=DeepKV.from_scratch(config(),arm,consumer=2,deep_target=4,
                        checkpoint_layers=checkpoint,checkpoint_lm=checkpoint,checkpoint_aux=checkpoint,lm_chunk=3)
                    if model.aux is not None:
                        with torch.no_grad():model.aux.out.weight.fill_(.03)
                    trainer=DeepKVTrainer(model=model,args=TrainingArguments(output_dir=folder,use_cpu=True,report_to=[]))
                    saved=copy.deepcopy(model.state_dict())
                    rng=torch.get_rng_state().clone()
                    reference=trainer.compute_loss(model,batch);reference.backward()
                    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
                    model.zero_grad(set_to_none=True)
                    audit=SDPAAudit(model)
                    with audit.observe():
                        loss=trainer.compute_loss(model,batch)
                        loss.register_hook(audit.backward_marker)
                        loss.backward()
                    torch.testing.assert_close(loss,reference,rtol=0,atol=0)
                    for n,p in model.named_parameters():
                        torch.testing.assert_close(p,saved[n],rtol=0,atol=0)
                        if n in grads:torch.testing.assert_close(p.grad,grads[n],rtol=0,atol=0)
                    self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                    result=audit.result()
                    self.assertFalse(result['unattributed_calls'])
                    self.assertFalse(result['unattributed_backward'])
                    self.assertEqual(sum(c['stage']=='forward' and c['site'].startswith('backbone') for c in result['calls']),4)
                    if model.consumer_aware:
                        detached=result['sites']['auxiliary.detached_consumer']
                        self.assertEqual(detached['forward_calls'],1)
                        self.assertTrue(detached['backward_operators'])
                    if checkpoint:self.assertTrue(any(c['stage']=='recompute' for c in result['calls']))
                    model.eval()
                    evaluation=SDPAAudit(model)
                    with evaluation.observe(),torch.no_grad():trainer.compute_loss(model,batch)
                    self.assertFalse(evaluation.result()['unattributed_calls'])
                    self.assertFalse(any(c['backward_operators'] for c in evaluation.calls))

    def test_wrappers_restore_after_failure(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        model=DeepKV.from_scratch(config(),'B',consumer=2,deep_target=4)
        sdpa,hf,aux=F.scaled_dot_product_attention,ALL_ATTENTION_FUNCTIONS['sdpa'],model.aux.forward
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with SDPAAudit(model).observe():raise RuntimeError('intentional')
        self.assertIs(F.scaled_dot_product_attention,sdpa)
        self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'],hf)
        self.assertEqual(model.aux.forward,aux)


if __name__ == '__main__':unittest.main()
