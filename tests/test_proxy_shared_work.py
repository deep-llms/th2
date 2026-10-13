"""Exact regression checks for reuse of per-forward, gradient-free work."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deep_kv import ALL_PROXY_ARMS
from deep_kv.proxy_estimators import gated_prediction
from deep_kv.training import DeepKVTrainer
from tests.test_proxy_screen24 import model, active, batch, objective


def legacy_consumer(layer, attention, hidden, prediction, target, alpha, mask, rotary, valid):
    # Previous implementation: recompute source RMS for both local responses.
    class Block(nn.Module):
        def __init__(self):
            super().__init__();self.layer=layer
        def forward(self, hidden, message, alpha, mask, rotary):
            with torch.autocast(hidden.device.type,enabled=False):
                scale=(hidden.float().square().mean(-1,keepdim=True)+1e-6).sqrt()
            injected=hidden+(gated_prediction(message,alpha,hidden)*scale).to(hidden.dtype)
            u=self.layer.input_layernorm(injected)
            attended=attention(self.layer,u,None,mask,rotary)
            residual=injected+self.layer.self_attn.o_proj(attended.reshape(*u.shape[:-1],-1).contiguous())
            return residual+self.layer.mlp(self.layer.post_attention_layernorm(residual))
    consumer=Block()
    frozen={name:p.detach() for name,p in consumer.named_parameters()}
    frozen.update({name:b.detach() for name,b in consumer.named_buffers()})
    with torch.no_grad():
        teacher=torch.func.functional_call(consumer,frozen,(hidden.detach(),target.detach(),alpha.detach(),mask,rotary))
    student=torch.func.functional_call(consumer,frozen,(hidden.detach(),prediction,alpha.detach(),mask,rotary))
    with torch.autocast(hidden.device.type,enabled=False):
        error=(student.float()-teacher.float()).square().mean(-1)
        scale=teacher.float().square().mean(-1)+1e-6
        return ((error/scale)*valid).sum(-1)[:,None]


class SharedWorkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_context_avoids_unused_defaults_and_preserves_fallback(self):
        ctx=batch()
        inputs=dict(input_ids=ctx.input_ids,attention_mask=ctx.valid,position_ids=ctx.position_ids,
                    segments=ctx.segments,labels=ctx.labels)
        with patch('torch.ones_like',side_effect=AssertionError('unused mask')), \
             patch('torch.arange',side_effect=AssertionError('unused positions')):
            actual=DeepKVTrainer.context(inputs)
        self.assertIs(actual.valid,ctx.valid)
        self.assertIs(actual.position_ids,ctx.position_ids)
        default=DeepKVTrainer.context(dict(input_ids=ctx.input_ids))
        torch.testing.assert_close(default.valid,torch.ones_like(ctx.input_ids,dtype=torch.bool))
        torch.testing.assert_close(default.position_ids,torch.arange(8).expand_as(ctx.input_ids))

    def test_all_proxy_outputs_gradients_and_adam_state_exact(self):
        # Both precision modes, active delayed arms, nonzero extension adapters,
        # and checkpoint replay. Reference repeats the old per-site count/RMS.
        for arm in ('A',)+ALL_PROXY_ARMS:
            for bf16 in (False,True):
                with self.subTest(arm=arm,bf16=bf16):
                    actual=model(arm,checkpoint=True);active(actual)
                    reference=copy.deepcopy(actual)
                    original=reference.layer_cosines
                    def repeat_counts(*args,**kw):
                        kw.pop('valid_counts',None)
                        return original(*args,**kw)
                    optimizers=[torch.optim.AdamW(m.parameters(),lr=3e-4) for m in (actual,reference)]
                    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
                        out=actual(batch(),collect_target_statistics=True)
                    objective(actual,out).backward()
                    with patch.object(reference,'layer_cosines',side_effect=repeat_counts), \
                         patch('deep_kv.proxy_estimators.consumer_effect_loss',legacy_consumer):
                        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
                            expected=reference(batch(),collect_target_statistics=True)
                        objective(reference,expected).backward()
                    for key in out:
                        torch.testing.assert_close(out[key],expected[key],rtol=0,atol=0,msg=arm+'/'+key)
                    for (name,p),(_,q) in zip(actual.named_parameters(),reference.named_parameters()):
                        self.assertEqual(p.grad is None,q.grad is None,name)
                        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=arm+'/'+name)
                    for optimizer in optimizers:optimizer.step()
                    for name,value in actual.state_dict().items():
                        torch.testing.assert_close(value,reference.state_dict()[name],rtol=0,atol=0,msg=arm+'/'+name)
                    for key,state in optimizers[0].state_dict()['state'].items():
                        for name,value in state.items():
                            torch.testing.assert_close(value,optimizers[1].state_dict()['state'][key][name],rtol=0,atol=0)

    def test_shared_counts_are_forward_local_and_not_used_at_inference(self):
        m=model('P6-iso');seen=[];original=m.layer_cosines
        def observe(*a,**kw):seen.append(kw['valid_counts']);return original(*a,**kw)
        with patch.object(m,'layer_cosines',side_effect=observe):
            m(batch());first=seen.copy();seen.clear();m(batch());second=seen.copy();seen.clear()
            m.hidden_states(batch());self.assertFalse(seen)
        self.assertEqual(len(first),len(m.layers))
        self.assertTrue(all(x is first[0] for x in first))
        self.assertTrue(all(x is second[0] for x in second))
        self.assertIsNot(first[0],second[0])


if __name__=='__main__':unittest.main()
