"""Independent attention oracle and live-target tests, not a CUDA speed claim."""
import copy
import unittest
from unittest.mock import patch

import torch

from scripts.p7_speed_trial import attention_lse, merge_attention, implementation
from deep_kv.proxy_memory import flash_joint_attention
from tests.test_proxy_memory import model, objective
from tests.test_proxy_heads import batch


def reference_kernel(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                     max_seqlen_k, causal, softmax_scale, return_lse=False):
    assert causal and torch.equal(cu_seqlens_q, cu_seqlens_k)
    outputs, lses = [], []
    for start, end in zip(cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist()):
        query = q[start:end].transpose(0, 1)
        keys = k[start:end].transpose(0, 1).repeat_interleave(q.shape[1]//k.shape[1], 0)
        values = v[start:end].transpose(0, 1).repeat_interleave(q.shape[1]//v.shape[1], 0)
        score = (query @ keys.transpose(-1, -2))*softmax_scale
        score = score.masked_fill(torch.ones(end-start, end-start, dtype=torch.bool).triu(1), -torch.inf)
        outputs.append((score.softmax(-1) @ values).transpose(0, 1))
        lses.append(score.logsumexp(-1))
    return torch.cat(outputs), torch.cat(lses, dim=1) if return_lse else None


class P7SpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_joint_output_and_all_five_gradients(self):
        torch.manual_seed(33)
        # Two packed rows, including singleton documents and row boundaries.
        layout = (torch.tensor([0, 1, 4, 7, 9, 13, 14], dtype=torch.int32), 4)
        tensors = [torch.randn(2, h, 7, 8, requires_grad=True) for h in (4, 2, 2, 2, 2)]
        q, k, v, kp, vp = tensors
        reference = flash_joint_attention(q, k, v, kp, vp, layout, .3, reference_kernel)
        n, nl = attention_lse(q, k, v, layout, .3, reference_kernel)
        p, pl = attention_lse(q, kp, vp, layout, .3, reference_kernel)
        merged = merge_attention(n, p, nl, pl)
        torch.testing.assert_close(reference, merged, rtol=2e-6, atol=5e-7)
        adjoint = torch.randn_like(reference)
        gradients = torch.autograd.grad((reference*adjoint).sum(), tensors, retain_graph=True)
        other = torch.autograd.grad((merged*adjoint).sum(), tensors, retain_graph=True)
        for a, b in zip(gradients, other): torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        # An incorrect detached-LSE implementation must NOT pass this test.
        broken = merge_attention(n, p, nl.detach(), pl.detach())
        bad = torch.autograd.grad((broken*adjoint).sum(), q)[0]
        self.assertGreater((bad-gradients[0]).norm().item(), .1)

    def test_causal_and_document_gradient_isolation(self):
        layout = (torch.tensor([0, 3, 4, 8], dtype=torch.int32), 4)
        tensors = [torch.randn(1, h, 8, 8, requires_grad=True) for h in (4, 2, 2, 2, 2)]
        q, k, v, kp, vp = tensors
        n, nl = attention_lse(q, k, v, layout, .3, reference_kernel)
        p, pl = attention_lse(q, kp, vp, layout, .3, reference_kernel)
        out = merge_attention(n, p, nl, pl)
        gradients = torch.autograd.grad(out[:, 5].square().sum(), tensors)
        for gradient in gradients:
            self.assertEqual(gradient[:, :, :4].abs().max().item(), 0)
            self.assertEqual(gradient[:, :, 6:].abs().max().item(), 0)

    def test_complete_model_objectives_buffers_and_optimizer(self):
        with patch('deep_kv.fa4.load_kernel', return_value=(reference_kernel, {})):
            for checkpoint in (False, True):
                for mode in ('cosine', 'lse', 'both'):
                    captures = []
                    for candidate in ('eager', mode):
                        with implementation(candidate, backend='aot_eager'):
                            m = model('P7-simple-targetgrad', 'fa4', checkpoint)
                            optimizer = torch.optim.AdamW(m.parameters(), lr=.0003)
                            out = m(batch(), collect_target_statistics=True)
                            # Both independent paths matter: auxiliary -> deep
                            # backbone remains live, LM -> predictor stays blocked.
                            named = dict(m.named_parameters())
                            names = [n for n in named if '.w1.' in n or '.mlp.' in n]
                            grads = {}
                            for label, loss in (('lm', out['lm_sum']/out['lm_count']),
                                                ('aux', out['aux_sum']/out['aux_count'])):
                                values = torch.autograd.grad(loss, [named[n] for n in names], allow_unused=True, retain_graph=True)
                                grads.update({label+'/'+n: v.detach().clone() for n, v in zip(names, values) if v is not None})
                            self.assertTrue(any(n.startswith('aux/backbone.') and v.abs().sum()>0 for n, v in grads.items()))
                            self.assertFalse(any(n.startswith('lm/heads.') for n in grads))
                            objective(out).backward()
                            grads.update({n: p.grad.clone() for n, p in named.items() if p.grad is not None})
                            optimizer.step()
                            m.update_statistics(out['center_sums'], out['center_squares'], out['center_counts'])
                            captures.append((out, grads, copy.deepcopy(m.state_dict())))
                    for a, b in zip(*captures):
                        self.assertEqual(a.keys(), b.keys())
                        for key in a:
                            torch.testing.assert_close(a[key], b[key], rtol=4e-4, atol=3e-6, msg=f'{mode}/{checkpoint}/{key}')


if __name__ == '__main__': unittest.main()
