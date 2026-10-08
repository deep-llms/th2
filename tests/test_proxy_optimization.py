"""Reference equivalence for forward-local target caches and proxy-key RoPE."""
import copy
from contextlib import contextmanager
import io
import unittest
from unittest.mock import patch

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from deep_kv import P6_VARIANTS, SIMPLE_MEMORY_ARMS
from deep_kv.model import Context
from deep_kv.proxy_memory import MemoryHead, flash_joint_attention
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


def legacy_estimate(self, u, plan):
    x = self.w_in(u.detach())
    c = x*self.conv[0]
    for lag, same in enumerate(plan.conv_masks, start=1):
        c = c + torch.nn.functional.pad(x[:, :-lag]*self.conv[lag]*same, (0, 0, lag, 0))
    if plan.ems is not None:
        c = c + (plan.ems.apply(x, 1)+plan.ems.apply(x, 2)).to(c.dtype)
    return self.w_out(torch.nn.functional.silu(c))


@contextmanager
def legacy_operations(m):
    def uncached_joint(*args, **kwargs):
        kwargs.pop('doubled_layout', None)
        return flash_joint_attention(*args, **kwargs)
    with patch.object(m, 'normalize_target', side_effect=lambda *a, **kw: legacy_normalize(m, *a, **kw)), \
         patch.object(MemoryHead, 'entries', legacy_entries), \
         patch.object(MemoryHead, 'estimate', legacy_estimate), \
         patch('deep_kv.proxy_memory.flash_joint_attention', side_effect=uncached_joint):
        yield


class ProxyOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        kernel = patch('deep_kv.fa4.load_kernel', return_value=(reference_kernel, {'version': 'CPU-oracle'}))
        kernel.start(); self.addCleanup(kernel.stop)

    def test_full_model_reference_outputs_gradients_and_buffers(self):
        arms = ('P6', 'P6-iso', *P6_VARIANTS, *SIMPLE_MEMORY_ARMS, 'P7', 'P7-kq', 'P7-ems', 'P7-mlp')
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
                        with legacy_operations(reference):
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

    def assert_nested_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for x, y in zip(actual, expected):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(actual, expected)

    def test_optimizer_statistics_trajectory_and_reload_match_legacy(self):
        arms = ('P6-iso-weighted', 'P6-iso-layernorm', 'P7-simple-sparse', 'P7-simple-short')
        for arm in arms:
            for backend in ('sdpa', 'fa4'):
                for bf16 in (False, True):
                    with self.subTest(arm=arm, backend=backend, bf16=bf16):
                        m = model(arm, backend, checkpoint=True)
                        reference = copy.deepcopy(m)
                        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
                        ref_opt = torch.optim.AdamW(reference.parameters(), lr=3e-4)
                        # Exercise actual two-pass initialization before optimizer updates.
                        for phase in ('mean', 'variance'):
                            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                                actual = m(batch(), compute_auxiliary_losses=False,
                                           collect_target_statistics=True, statistics_mode=phase)
                                with legacy_operations(reference):
                                    expected = reference(batch(), compute_auxiliary_losses=False,
                                                         collect_target_statistics=True, statistics_mode=phase)
                            self.assert_nested_equal(actual, expected)
                            for current, out in ((m, actual), (reference, expected)):
                                current.update_statistics(out['center_sums'], out['center_squares'],
                                                          out['center_counts'], initialize=phase)
                        for step in range(3):
                            opt.zero_grad(set_to_none=True); ref_opt.zero_grad(set_to_none=True)
                            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                                actual = m(batch(), collect_target_statistics=True)
                            objective(actual).backward()
                            with legacy_operations(reference):
                                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                                    expected = reference(batch(), collect_target_statistics=True)
                                objective(expected).backward()
                            self.assert_nested_equal(actual, expected)
                            self.assert_nested_equal([p.grad for p in m.parameters()],
                                                     [p.grad for p in reference.parameters()])
                            opt.step(); ref_opt.step()
                            for current, out in ((m, actual), (reference, expected)):
                                current.update_statistics(out['center_sums'], out['center_squares'], out['center_counts'])
                            self.assert_nested_equal(m.state_dict(), reference.state_dict())
                            self.assert_nested_equal(opt.state_dict(), ref_opt.state_dict())
                            if step == 1:
                                # Resume into a new model/optimizer; reference remains uninterrupted.
                                saved = io.BytesIO()
                                torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict()), saved)
                                saved.seek(0); state = torch.load(saved, weights_only=True)
                                m = model(arm, backend, checkpoint=True)
                                m.load_state_dict(state['model'], strict=True)
                                opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
                                opt.load_state_dict(state['optimizer'])

    def test_outstanding_forwards_with_different_documents_match_uncached_replay(self):
        # Single-token documents, unequal fragments, and a changed batch/sequence
        # shape must all keep their own FA4 layout until their backward completes.
        contexts = [batch()]
        for segments in (torch.tensor([[0, 1, 1, 2, 2, 2]]), torch.tensor([[0, 0, 0], [0, 1, 1]])):
            positions = torch.zeros_like(segments)
            for t in range(1, segments.shape[1]):
                positions[:, t] = torch.where(segments[:, t] == segments[:, t-1], positions[:, t-1]+1, 0)
            ids = torch.arange(segments.numel()).reshape_as(segments)+3
            contexts.append(Context(ids, torch.ones_like(ids, dtype=torch.bool), positions, segments))
        for arm in SIMPLE_MEMORY_ARMS:
            for bf16 in (False, True):
                with self.subTest(arm=arm, bf16=bf16):
                    m = model(arm, 'fa4', checkpoint=True)
                    reference = copy.deepcopy(m)
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                        actual = [m(ctx, collect_target_statistics=True) for ctx in contexts]
                    sum(objective(out) for out in reversed(actual)).backward()
                    with legacy_operations(reference):
                        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                            expected = [reference(ctx, collect_target_statistics=True) for ctx in contexts]
                        sum(objective(out) for out in reversed(expected)).backward()
                    self.assert_nested_equal(actual, expected)
                    self.assert_nested_equal([p.grad for p in m.parameters()],
                                             [p.grad for p in reference.parameters()])
