"""Bottleneck mechanism, gradient boundaries, and native Trainer integration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import TrainingArguments, default_data_collator

from deep_kv import BOTTLENECK_ARMS
from deep_kv.model import Context, normalize_code
from deep_kv.training import DeepKVTrainer, summarize_statistics
from deep_kv.__main__ import jobs
from deep_kv.report import report
from test_deep_kv import model, context, parameter_hash
from test_train import fixture, invoke


def objective(m, out):
    return (out['lm_sum'] / out['lm_count'].clamp_min(1)
            + out['extractor_sum'] / out['extractor_count'].clamp_min(1)
            + m.code_loss_weight * out['align_sum'] / out['align_count'].clamp_min(1))


class BottleneckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matched_initialization_and_forward_isolation(self):
        models = [model(arm).eval() for arm in BOTTLENECK_ARMS]
        for m in models:
            self.assertEqual(parameter_hash(m.backbone), parameter_hash(models[0].backbone))
            self.assertEqual(parameter_hash(m.aux), parameter_hash(models[0].aux))
            self.assertEqual(parameter_hash(m.extractor), parameter_hash(models[0].extractor))
            self.assertIsNot(m.readout.weight, m.backbone.lm_head.weight)
            self.assertEqual(m.aux.predictor.out_features, 128)
            self.assertEqual(m.aux.k.in_features, 128)
            self.assertEqual(m.readout.in_features, 32 if m.consumer_aware else 128)
            self.assertEqual(m.parameter_counts()['inference'], models[0].parameter_counts()['inference'])
        for a, b in ((0, 1), (2, 3)):
            self.assertEqual(parameter_hash(models[a]), parameter_hash(models[b]))
            self.assertEqual((models[a].code_loss_weight, models[b].code_loss_weight), (0, .3))
        logits = []
        for m in models:
            with torch.no_grad():
                m.aux.out.weight.fill_(.02)
                before = m.backbone.lm_head(m.hidden_states(context())[0])
                normal = m(context())['lm_sum']
                self.assertEqual(normal, m(context(), compute_auxiliary_losses=False)['lm_sum'])
                m.extractor.weight.fill_(100)
                m.readout.weight.fill_(-100)
                after = m.backbone.lm_head(m.hidden_states(context())[0])
                torch.testing.assert_close(before, after, atol=0, rtol=0)
                # Removing training-only modules still permits the real path.
                del m.extractor, m.readout
                self.assertEqual(normal, m(context(), compute_auxiliary_losses=False)['lm_sum'])
                logits.append(before)
        for value in logits[1:]:
            torch.testing.assert_close(value, logits[0], atol=0, rtol=0)

    def test_gradient_boundaries(self):
        for arm in BOTTLENECK_ARMS:
            for checkpoint in (False, True):
                m = model(arm, checkpoint=checkpoint).train()
                # Exercise every detached functional parameter with an open output.
                with torch.no_grad():
                    m.aux.out.weight.normal_(std=.1)
                for term in ('extractor_sum', 'align_sum', 'lm_sum'):
                    m.zero_grad(set_to_none=True)
                    m(context())[term].backward()
                    grads = {n for n, p in m.named_parameters()
                             if p.grad is not None and bool(p.grad.count_nonzero())}
                    self.assertTrue(grads)
                    if term == 'extractor_sum':
                        self.assertEqual(grads, {'extractor.weight', 'readout.weight'})
                    elif term == 'align_sum':
                        self.assertIn('aux.predictor.weight', grads)
                        self.assertTrue(any(n.startswith('backbone.') for n in grads))
                        self.assertFalse(any(n.startswith(('extractor.', 'readout.')) for n in grads))
                        self.assertEqual({n for n in grads if n.startswith('aux.')}, {'aux.predictor.weight'})
                        self.assertFalse(any(n.startswith(('backbone.model.layers.2.', 'backbone.model.layers.3.')) for n in grads))
                    else:
                        self.assertFalse(any(n.startswith(('extractor.', 'readout.')) for n in grads))
                        self.assertTrue(all('aux.' + n in grads for n in
                                            ('predictor.weight', 'k.weight', 'v.weight', 'out.weight', 'k_norm.weight')))

    def test_zero_output_initialization_and_checkpoint_parity(self):
        for arm in BOTTLENECK_ARMS:
            m = model(arm).train()
            m(context())['extractor_sum'].backward()
            self.assertEqual(bool(m.extractor.weight.grad.count_nonzero()), not m.consumer_aware)
            m.zero_grad(set_to_none=True)
            m(context())['lm_sum'].backward()
            self.assertTrue(bool(m.aux.out.weight.grad.count_nonzero()))
            self.assertEqual(m.aux.predictor.weight.grad.count_nonzero(), 0)
            with torch.no_grad():
                m.aux.out.weight.normal_(std=.03)
            other = copy.deepcopy(m)
            m.checkpoint_layers = m.checkpoint_lm = m.checkpoint_aux = False
            other.checkpoint_layers = other.checkpoint_lm = other.checkpoint_aux = True
            for value in (m, other):
                value.zero_grad(set_to_none=True)
                objective(value, value(context())).backward()
            for (n, p), (_, q) in zip(m.named_parameters(), other.named_parameters()):
                self.assertIsNotNone(p.grad, n)
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)

    def test_locations_single_pass_and_loss_masks(self):
        for arm in BOTTLENECK_ARMS:
            m = model(arm).eval()
            captured, calls = {}, [0] * 4
            handles = []
            for i, block in enumerate(m.backbone.model.layers):
                def observe(module, args, index=i):
                    captured[index] = args[0].detach().clone()
                    calls[index] += 1
                handles.append(block.input_layernorm.register_forward_pre_hook(observe))
            hidden, predicted, target = m.hidden_states(context())
            for handle in handles:
                handle.remove()
            self.assertEqual(calls, [1] * 4)
            torch.testing.assert_close(target[0], captured[3], atol=0, rtol=0)
            self.assertFalse(target[0].requires_grad)
            if m.consumer_aware:
                torch.testing.assert_close(predicted[2], captured[1], atol=0, rtol=0)
                self.assertFalse(predicted[1].requires_grad)
            out = m(context())
            self.assertEqual(out['lm_count'], 6)
            self.assertEqual(out['extractor_count'], 3 if m.consumer_aware else 6)
            self.assertEqual(out['align_count'], 9)
            metric = summarize_statistics(out['statistics'].numpy(), arm=arm, evaluation=True)
            self.assertAlmostEqual(metric['objective'], objective(m, out).item(), places=5)
            if m.consumer_aware:
                self.assertAlmostEqual(metric['message_ce_gain'], 0, places=6)
            batch = context()
            batch.labels = batch.input_ids.clone()
            batch.labels[0, 2] = -100
            masked = m(batch)
            self.assertEqual(masked['lm_count'], 5)
            self.assertEqual(masked['extractor_count'], 2 if m.consumer_aware else 5)
            self.assertEqual(masked['align_count'], 9)

    def test_causal_decoder_and_fp32_losses(self):
        m = model('Consumer-Aware-Align').train()
        with torch.no_grad():
            m.aux.out.weight.normal_(std=.1)
        batch = context()
        code = normalize_code(torch.randn(2, 6, 128)).requires_grad_()
        q = torch.randn(2, 4, 6, 8)
        rotary = m.backbone.model.rotary_emb(torch.empty(2, 6, 32), batch.position_ids)
        for detached in (False, True):
            def decode(z):
                if detached:
                    return torch.func.functional_call(m.aux, {n: p.detach() for n, p in m.aux.named_parameters()},
                        (z, q, rotary, batch.allowed()), {'decode_only': True})[0]
                return m.aux(z, q, rotary, batch.allowed(), decode_only=True)[0]
            correction = decode(code)
            self.assertTrue(torch.isfinite(correction).all())
            self.assertEqual(correction[0, [0, 3]].count_nonzero(), 0)
            self.assertEqual(correction[1, [0, 3, 4, 5]].count_nonzero(), 0)
            changed = code.detach().clone()
            changed[:, 2:] += 100
            torch.testing.assert_close(decode(changed)[:, :2], correction[:, :2], atol=0, rtol=0)
            grad = torch.autograd.grad(correction[0, 1].sum(), code)[0]
            self.assertGreater(grad[0, 0].abs().sum(), 0)
            self.assertEqual(grad[0, 1:].count_nonzero(), 0)
        for arm in BOTTLENECK_ARMS:
            m = model(arm)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                out = m(batch)
                loss = objective(m, out)
            for name in ('lm_sum', 'extractor_sum', 'align_sum'):
                self.assertEqual(out[name].dtype, torch.float32)
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        value = torch.tensor([[1e4, -1e4]], dtype=torch.bfloat16)
        expected = value.float() / (value.float().square().mean(-1, keepdim=True) + 1e-6).sqrt()
        torch.testing.assert_close(normalize_code(value), expected)
        # Length two has an LM/task target but no eligible consumer query:
        # t=0 has no past, and t=1 has no next-token label.
        short = Context(torch.tensor([[3, 5]]), torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0, 1]]))
        m = model('Consumer-Aware-Align')
        out = m(short)
        self.assertEqual(out['lm_count'], 1)
        self.assertEqual(out['extractor_count'], 0)
        self.assertEqual(out['extractor_sum'], 0)
        self.assertTrue(torch.isfinite(objective(m, out)))
        out['extractor_sum'].backward()
        self.assertEqual(m.extractor.weight.grad.count_nonzero(), 0)

    def test_native_trainer_accumulation_with_unequal_masks(self):
        batch = context()
        data = Dataset.from_dict({'input_ids': batch.input_ids.tolist(), 'attention_mask': batch.valid.tolist(),
                                  'position_ids': batch.position_ids.tolist(), 'segments': batch.segments.tolist()})
        for arm in BOTTLENECK_ARMS:
            states = []
            for micro, accumulation in ((2, 1), (1, 2)):
                with tempfile.TemporaryDirectory() as tmp:
                    m = model(arm)
                    with torch.no_grad():
                        m.aux.out.weight.fill_(.03)
                    args = TrainingArguments(output_dir=tmp, use_cpu=True, report_to='none', max_steps=1,
                        per_device_train_batch_size=micro, gradient_accumulation_steps=accumulation,
                        optim='sgd', learning_rate=.01, max_grad_norm=0, remove_unused_columns=False,
                        save_strategy='no', logging_strategy='no', disable_tqdm=True)
                    trainer = DeepKVTrainer(model=m, args=args, train_dataset=data,
                                            data_collator=default_data_collator)
                    trainer.train()
                    states.append(copy.deepcopy(m.state_dict()))
            for name in states[0]:
                torch.testing.assert_close(states[0][name], states[1][name], atol=2e-7, rtol=2e-6, msg=name)

    def test_train_entry_resume_report_and_sequential_queue(self):
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            for arm in BOTTLENECK_ARMS:
                invoke(root, {**config, 'arm': arm, 'output_dir': str(root / arm), 'stop_after': 2})
                invoke(root, {**config, 'arm': arm, 'output_dir': str(root / arm)})
                invoke(root, {**config, 'arm': arm, 'output_dir': str(root / ('full-' + arm))})
                resumed = load_file(root / arm / 'model.safetensors')
                full = load_file(root / ('full-' + arm) / 'model.safetensors')
                for name in full:
                    torch.testing.assert_close(resumed[name], full[name], atol=0, rtol=0)
                result = json.loads((root / arm / 'result.json').read_text())
                evaluation = result['evaluation']
                self.assertEqual(evaluation['eval_loss'], evaluation['eval_lm_loss'])
                self.assertEqual(evaluation['eval_rows'], 5)
                self.assertEqual(evaluation['eval_extractor_targets'], 5 * (6 if arm.startswith('Consumer') else 7))
                cost = result['training_cost']
                self.assertEqual(cost['optimizer_steps'], 1)
                self.assertGreater(cost['input_tokens_per_second'], 0)
                self.assertGreater(cost['parameters']['training_only'], 0)
            comparison = report(root, BOTTLENECK_ARMS)
            self.assertEqual(len(comparison['nll_differences']), 3)
            from scripts.stage_deep_kv_resume import stage
            receipt = stage({arm: str(root / arm) for arm in BOTTLENECK_ARMS}, root / 'staged', 3)
            self.assertEqual(set(receipt['arms']), set(BOTTLENECK_ARMS))
            recipe = root / 'recipe.json'
            recipe.write_text(json.dumps(config))
            queue = jobs(recipe, stop_after=3, arms=BOTTLENECK_ARMS)
            queue_path = root / 'jobs.json'
            queue_path.write_text(json.dumps(queue))
            load_jobs(queue_path)
            self.assertEqual(len(queue['jobs']), 5)
            for arm, job in zip(BOTTLENECK_ARMS, queue['jobs']):
                self.assertEqual(job['gpus'], list(range(8)))
                self.assertEqual(job['argv'][job['argv'].index('--arm') + 1], arm)


if __name__ == '__main__':
    unittest.main()
