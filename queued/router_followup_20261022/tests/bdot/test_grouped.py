"""Grouped dispatch contract; CPU substitutes only the CUDA GEMM primitive."""
import copy
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import torch
from test_model import model, batch
from bdot import grouped
from bdot.model import BDOTForCausalLM


def reference_mm(x, weights, offsets):
    starts = [0, *offsets.tolist()]
    assert len(weights) == len(offsets) and starts[-1] == len(x)
    assert all(b > a for a, b in zip(starts, starts[1:]))
    return torch.cat([x[a:b] @ w.T for a, b, w in zip(starts, starts[1:], weights)])


def enable_grouped(m):
    m.options = replace(m.options, expert_backend='grouped')
    m.config.bdot = m.options.to_dict()
    for step in m.routed_steps:
        step.options = m.options
    return m


class GroupedTests(unittest.TestCase):
    def test_linear_irregular_groups_and_empty_expert_momentum(self):
        layers = torch.nn.ModuleList([torch.nn.Linear(16, 24, bias=False) for _ in range(4)])
        other = copy.deepcopy(layers)
        x = torch.randn(13, 16, requires_grad=True); y = x.detach().clone().requires_grad_()
        offsets = torch.tensor([3, 4, 13], dtype=torch.int32)
        opt = torch.optim.AdamW(layers.parameters(), lr=.001, weight_decay=.1)
        for p in layers[1].parameters():
            opt.state[p] = dict(step=torch.tensor(5.), exp_avg=torch.ones_like(p), exp_avg_sq=torch.ones_like(p))
        unused = layers[1].weight.detach().clone()
        with patch.object(grouped, '_grouped_mm', side_effect=reference_mm) as kernel:
            a = grouped.linear(x, [layers[i] for i in (0,2,3)], offsets)
        b = torch.cat([other[i](part) for i, part in zip((0,2,3), y.split([3,1,9]))])
        torch.testing.assert_close(a,b)
        a.square().sum().backward(); b.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad)
        for p,q in zip(layers.parameters(),other.parameters()):
            if q.grad is None: self.assertIsNone(p.grad)
            else: torch.testing.assert_close(p.grad,q.grad)
        opt.step()
        torch.testing.assert_close(layers[1].weight,unused,rtol=0,atol=0)
        self.assertEqual(float(opt.state[layers[1].weight]['step']),5.)
        self.assertEqual(kernel.call_count,1)

    def test_models_routes_gradients_updates_and_checkpointing(self):
        variants = [dict(curriculum='A', progress=.05),dict(curriculum='A',progress=.2),
                    dict(curriculum='B',progress=.01),dict(curriculum='B',progress=.2),
                    dict(curriculum='B',progress=.01,reuse=False),
                    dict(curriculum='none',progress=1.,reuse=False),
                    dict(curriculum='none',progress=1.,gate='scale')]
        for variant in variants:
            progress=variant.pop('progress')
            for checkpointing in (False,True):
                with self.subTest(variant=variant,checkpointing=checkpointing):
                    a=model(default_prior=0.,**variant); b=enable_grouped(copy.deepcopy(a))
                    a.progress=b.progress=progress; a.capture_routes=b.capture_routes=True
                    if checkpointing:
                        b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                    with patch.object(grouped,'_grouped_mm',side_effect=reference_mm):
                        x=a(**batch());y=b(**batch())
                        torch.testing.assert_close(x.logits,y.logits,atol=1e-6,rtol=1e-5)
                        torch.testing.assert_close(a.last_routes,b.last_routes,atol=0,rtol=0)
                        x.loss.backward();y.loss.backward()
                    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
                        if p.grad is None:self.assertIsNone(q.grad,name)
                        else:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4,msg=name)
                    for m in (a,b):
                        torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.1).step();m.finish_update()
                    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
                        torch.testing.assert_close(p,q,atol=1e-5,rtol=1e-4,msg=name)
                    torch.testing.assert_close(a.balance_bias,b.balance_bias,atol=0,rtol=0)

    def test_grouped_checkpoint_loading_and_backend_guard(self):
        m=model(expert_backend='grouped',curriculum='none',default_prior=0.).eval()
        with tempfile.TemporaryDirectory() as path:
            m.save_pretrained(path)
            restored=BDOTForCausalLM.from_pretrained(path,attn_implementation='sdpa').eval()
            self.assertEqual(restored.options.expert_backend,'grouped')
            with patch.object(grouped,'_grouped_mm',side_effect=reference_mm):
                torch.testing.assert_close(m(**batch()).logits,restored(**batch()).logits,atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'CUDA BF16'):
            grouped._grouped_mm(torch.zeros(8,16),torch.zeros(2,24,16),torch.tensor([4,8],dtype=torch.int32))


if __name__=='__main__':unittest.main()
