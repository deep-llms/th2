"""Reference equivalence for forward-local target caches and proxy-key RoPE."""
import copy
import unittest
from unittest.mock import patch

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from deep_kv import P6_VARIANTS, SIMPLE_MEMORY_ARMS
from deep_kv.proxy_memory import MemoryHead
from tests.test_proxy_memory import model, batch, objective
from tests.test_fa4_baseline import reference_kernel


def legacy_normalize(m, value, key, *, return_clipped=False, **unused):
    with torch.no_grad(), torch.autocast(value.device.type, enabled=False):
        index = m.mean_index[key]
        variance = m.sigma2[index]
        floor = m.settings.variance_floor * variance.median()
        value = (value.detach().float() - m.mu[index]) * torch.rsqrt(variance.clamp_min(floor) + 1e-6)
        normalized = value.clamp(-m.settings.target_clip, m.settings.target_clip)
        return (normalized, (value.abs() > m.settings.target_clip).sum()) if return_clipped else normalized


def legacy_entries(self, prediction, rotary, native_values):
    with torch.autocast(prediction.device.type, enabled=False):
        value = (prediction if getattr(self, 'task_finetuning', False) else prediction.detach()).float()
        p = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(prediction.dtype)
    shape = (*p.shape[:-1], 2, -1)
    k = self.k_norm(self.k_proj(p).view(shape)).transpose(1, 2)
    k, _ = apply_rotary_pos_emb(k, k, *rotary)
    v = native_values if self.v_proj is None else self.v_proj(p).view(shape).transpose(1, 2)
    return k, v


class ProxyOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        kernel = patch('deep_kv.fa4.load_kernel', return_value=(reference_kernel, {'version': 'CPU-oracle'}))
        kernel.start(); self.addCleanup(kernel.stop)

    def test_full_model_reference_outputs_gradients_and_buffers(self):
        arms = ('P6', 'P6-iso', *P6_VARIANTS, *SIMPLE_MEMORY_ARMS, 'P7')
        for arm in arms:
            for backend in ('sdpa', 'fa4'):
                for bf16 in (False, True):
                    with self.subTest(arm=arm, backend=backend, bf16=bf16):
                        m = model(arm, backend, checkpoint=True)
                        with torch.no_grad():
                            m.mu.copy_(torch.linspace(-.2, .2, m.mu.numel()).reshape_as(m.mu))
                            m.sigma2.copy_(torch.linspace(0, .03, m.sigma2.numel()).reshape_as(m.sigma2))
                        reference = copy.deepcopy(m)
                        state = {k: v.clone() for k, v in m.state_dict().items()}
                        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                            actual = m(batch(), collect_target_statistics=True)
                        objective(actual).backward()
                        with patch.object(reference, 'normalize_target', side_effect=lambda *a, **kw: legacy_normalize(reference, *a, **kw)), \
                             patch.object(MemoryHead, 'entries', legacy_entries):
                            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                                expected = reference(batch(), collect_target_statistics=True)
                            objective(expected).backward()
                        for key in actual:
                            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
                        for (name, p), (_, q) in zip(m.named_parameters(), reference.named_parameters()):
                            self.assertEqual(p.grad is None, q.grad is None, name)
                            if p.grad is not None:
                                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
                        for key, value in m.state_dict().items():
                            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_normalization_cache_is_forward_local_and_skips_unused_work(self):
        for arm in ('P6-iso-weighted', 'P6-iso-layernorm', 'P7-simple-short'):
            m = model(arm)
            original = m.normalize_target
            seen = []
            def observe(value, key, **kw):
                seen.append(kw)
                return original(value, key, **kw)
            with patch.object(m, 'normalize_target', side_effect=observe):
                first = m(batch(), collect_target_statistics=True)
                self.assertTrue(all(x['centered'] is not None and x['return_clipped'] for x in seen))
                scales = [x['inverse_scale'].clone() for x in seen]
                seen.clear()
                with torch.no_grad(): m.mu.add_(.3); m.sigma2.mul_(2)
                second = m(batch())
                self.assertTrue(all(x['centered'] is None and not x['return_clipped'] for x in seen))
                self.assertFalse(torch.equal(scales[0], seen[0]['inverse_scale']))
                self.assertFalse(torch.equal(first['aux_sum'], second['aux_sum']))
            # Bootstrap and inference do not need normalization scales.
            with patch.object(m, 'normalize_target', side_effect=AssertionError('unneeded normalization')):
                for phase in ('mean', 'variance'):
                    m(batch(), compute_auxiliary_losses=False, collect_target_statistics=True, statistics_mode=phase)
                m.hidden_states(batch())

    def test_fa4_doubled_layout_shared_in_forward_and_replay(self):
        for arm in SIMPLE_MEMORY_ARMS:
            m = model(arm, 'fa4', checkpoint=True)
            native, doubled = [], []
            def observe(q, k, v, layout):
                (doubled if q.shape[2] == 16 else native).append(layout)
            m._fa4_observer = observe
            objective(m(batch())).backward()
            self.assertTrue(doubled)
            self.assertTrue(all(x is doubled[0] for x in doubled))
            torch.testing.assert_close(doubled[0][0], native[0][0] * 2, rtol=0, atol=0)
            self.assertEqual(doubled[0][1], native[0][1] * 2)
            previous = doubled[0]
            doubled.clear(); native.clear()
            m(batch())
            self.assertIsNot(previous, doubled[0])
