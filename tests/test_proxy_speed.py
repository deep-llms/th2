"""Training-state and diagnostic contracts for execution-only proxy changes."""
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset
from transformers import TrainingArguments

from deep_kv.packing import isolated_data_collator
from deep_kv.proxy_memory import MemoryPlan
from deep_kv.proxy_training import ProxyCallback, ProxyTrainer
from tests.test_proxy_memory import model, batch, objective
from tests.test_proxy_optimization import legacy_estimate


class ProxySpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_convolution_masks_reuse_and_reference_gradients(self):
        for bf16, parameter_dtype in ((False, torch.float32), (True, torch.float32), (False, torch.bfloat16)):
            for length in (1, 2, 3, 8, 17):
                with self.subTest(bf16=bf16, parameter_dtype=parameter_dtype, length=length):
                    torch.manual_seed(113+length)
                    head = model('P7').heads['2'].to(dtype=parameter_dtype)
                    reference = copy.deepcopy(head)
                    docs = torch.arange(length)[None].expand(2, -1)//3
                    plan = MemoryPlan(docs, sample_queries=False)
                    source = torch.randn(2, length, 32, dtype=parameter_dtype, requires_grad=True)
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                        actual = head.estimate(source, plan)
                        expected = legacy_estimate(reference, source, plan)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    actual.float().square().sum().backward()
                    expected.float().square().sum().backward()
                    self.assertIsNone(source.grad)
                    for p, q in zip(head.parameters(), reference.parameters()):
                        self.assertEqual(p.grad is None, q.grad is None)
                        if p.grad is not None:
                            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                    for dtype in (torch.float32, torch.bfloat16):
                        masks = plan.convolution_masks(dtype)
                        self.assertIs(masks, plan.convolution_masks(dtype))
                        for value, original in zip(masks, plan.conv_masks):
                            torch.testing.assert_close(value, original.to(dtype), rtol=0, atol=0)
                    if plan.conv_masks:
                        self.assertIsNot(plan.convolution_masks(torch.float32),
                                         MemoryPlan(docs, sample_queries=False).convolution_masks(torch.float32))

    def test_clip_switch_preserves_losses_gradients_and_moments(self):
        for arm in ('P6-iso-weighted', 'P6-iso-layernorm', 'P7-simple', 'P7'):
            m = model(arm, checkpoint=True)
            with torch.no_grad():
                m.mu.fill_(10.)  # Ensure the reference actually observes clipping.
                m.sigma2.fill_(.01)
            expected = m(batch(), collect_target_statistics=True)
            objective(expected).backward()
            grads = [None if p.grad is None else p.grad.clone() for p in m.parameters()]
            m.zero_grad(set_to_none=True)
            original = m.normalize_target
            flags = []
            def observe(*args, **kwargs):
                flags.append(kwargs['return_clipped'])
                return original(*args, **kwargs)
            with patch.object(m, 'normalize_target', side_effect=observe):
                actual = m(batch(), collect_target_statistics=True, collect_clip_statistics=False)
            self.assertTrue(flags); self.assertFalse(any(flags))
            self.assertGreater(expected['clip_counts'].sum(), 0)
            self.assertEqual(actual['clip_counts'].count_nonzero(), 0)
            for key in actual.keys()-{'clip_counts'}:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            objective(actual).backward()
            for p, grad in zip(m.parameters(), grads):
                if grad is None:self.assertIsNone(p.grad)
                else:torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
            # Statistics-only microbatches still collect every target, even when
            # normalization/clipping is not needed for a zero-weight auxiliary.
            with patch.object(m, 'normalize_target', side_effect=AssertionError('unused normalization')):
                stats = m(batch(), compute_auxiliary_losses=False, collect_target_statistics=True,
                          collect_clip_statistics=False)
            for key in ('center_sums', 'center_squares', 'center_counts'):
                torch.testing.assert_close(stats[key], expected[key], rtol=0, atol=0)

    def test_trainer_clipping_schedule_accumulation_and_legacy_state(self):
        ctx = batch()
        row = dict(input_ids=ctx.input_ids[0].tolist(), labels=ctx.labels[0].tolist(),
                   attention_mask=[1]*8, segments=ctx.segments[0].tolist())
        dataset = Dataset.from_list([row]*8)
        for first, strategy in ((False, 'steps'), (True, 'steps'), (False, 'epoch')):
            with tempfile.TemporaryDirectory() as tmp:
                snapshots, flags, histories, optimizers = [], [], [], []
                for force_legacy in (False, True):
                    m = model('P6-iso-weighted', checkpoint=True)
                    args = TrainingArguments(output_dir=tmp, use_cpu=True, report_to=[], max_steps=4,
                        learning_rate=3e-4, per_device_train_batch_size=1, gradient_accumulation_steps=2,
                        logging_steps=3, logging_first_step=first, logging_strategy=strategy, save_strategy='no',
                        remove_unused_columns=False, disable_tqdm=True)
                    callback = ProxyCallback(4, 16)
                    trainer = ProxyTrainer(model=m, args=args, train_dataset=dataset,
                        data_collator=isolated_data_collator, callbacks=[callback])
                    callback.trainer = trainer
                    forward = m.forward
                    seen = []
                    def observe(*a, **kw):
                        if kw.get('statistics_mode') is None:
                            seen.append(kw['collect_clip_statistics'])
                            if force_legacy:kw['collect_clip_statistics'] = True
                        return forward(*a, **kw)
                    with patch.object(m, 'forward', side_effect=observe):
                        trainer.train()
                    snapshots.append(copy.deepcopy(m.state_dict()))
                    optimizers.append(copy.deepcopy(trainer.optimizer.state_dict()))
                    flags.append(seen)
                    histories.append(trainer.state.log_history)
                    self.assertEqual('clip_fraction' in trainer.normalization_metrics, strategy == 'epoch')
                self.assertEqual(flags[0], [True]*8 if strategy == 'epoch' else [first]*2+[False]*2+[True]*2+[False]*2)
                for key, value in snapshots[0].items():
                    torch.testing.assert_close(value, snapshots[1][key], rtol=0, atol=0)
                for key, state in optimizers[0]['state'].items():
                    for name, value in state.items():
                        torch.testing.assert_close(value, optimizers[1]['state'][key][name], rtol=0, atol=0)
                for a, b in zip(histories[0], histories[1]):
                    if 'loss' not in a:continue
                    self.assertEqual(a['loss'], b['loss'])
                    for key in a:
                        if key.endswith('_clip_fraction'):
                            self.assertEqual(a[key], b[key])
                    self.assertTrue(any(key.endswith('_clip_fraction') for key in a))
