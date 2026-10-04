"""Bottleneck mechanism, gradient boundaries, and native Trainer integration."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F
from datasets import Dataset
from safetensors.torch import load_file
from transformers import TrainingArguments, default_data_collator

from deep_kv import BOTTLENECK_ARMS
from deep_kv.model import Context, DeepKV, normalize_code
from deep_kv.training import DeepKVTrainer, summarize_statistics
from deep_kv.__main__ import jobs
from deep_kv.report import report
from tests.test_deep_kv import model, context, parameter_hash, config
from tests.test_train import fixture, invoke


def objective(m, out):
    return (out['lm_sum'] / out['lm_count'].clamp_min(1)
            + out['extractor_sum'] / out['extractor_count'].clamp_min(1)
            + m.code_loss_weight * out['align_sum'] / out['align_count'].clamp_min(1))


class BottleneckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_literal_reference_at_production_hook_indices(self):
        # Qwen3-0.6B has query width != residual width. Preserve that ratio in a
        # tiny 22-block model and exercise the actual 1-based hooks 5 and 21.
        cfg = config()
        cfg.num_hidden_layers, cfg.head_dim = 22, 16
        cfg.layer_types = ['full_attention'] * cfg.num_hidden_layers
        batch = context()
        labels = batch.input_ids[:, 1:].clone()
        eligible = batch.valid[:, :-1] & batch.valid[:, 1:]
        eligible &= batch.segments[:, :-1].eq(batch.segments[:, 1:])
        labels.masked_fill_(~eligible, -100)
        n = batch.input_ids.shape[1]
        strict = (torch.arange(n)[None, :] < torch.arange(n)[:, None])[None, None]
        strict = strict & batch.valid[:, None, :, None] & batch.valid[:, None, None, :]
        strict &= batch.segments[:, None, :, None].eq(batch.segments[:, None, None, :])

        def rms(x, epsilon):
            return x.float() / (x.float().square().mean(-1, keepdim=True) + epsilon).sqrt()

        def ce(head, x, y):
            logits = F.linear(x, head.weight).float()
            return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   y.flatten(), reduction='sum', ignore_index=-100)

        for arm in BOTTLENECK_ARMS:
            actual = DeepKV.from_scratch(cfg, arm, checkpoint_layers=False, lm_chunk=3)
            with torch.no_grad():
                actual.aux.out.weight.normal_(std=.04)
            reference = copy.deepcopy(actual)
            out = actual(batch)
            objective(actual, out).backward()
            captured = {}
            def capture(index):
                def hook(module, args, output):
                    captured[index] = (args[0], output)
                return hook
            handles = [reference.backbone.model.layers[i].input_layernorm.register_forward_hook(capture(i))
                       for i in (4, 20)]
            hidden, predicted, target = reference.hidden_states(batch)
            for handle in handles:
                handle.remove()
            torch.testing.assert_close(target[0], captured[20][0], atol=0, rtol=0)
            torch.testing.assert_close(predicted[0], rms(reference.aux.predictor(captured[4][1]), 1e-6))
            code = rms(reference.extractor(captured[20][0].detach()), 1e-6)
            if reference.consumer_aware:
                torch.testing.assert_close(predicted[2], captured[4][0], atol=0, rtol=0)
                head = reference.backbone.model.layers[4].self_attn
                q = head.q_proj(captured[4][1]).view(2, n, 4, 16)
                q = rms(q, head.q_norm.variance_epsilon) * head.q_norm.weight
                q = q.transpose(1, 2).detach()
                cos, sin = target[1]
                def rotate(x):
                    halves = torch.cat((-x[..., 8:], x[..., :8]), dim=-1)
                    return x * cos[:, None] + halves * sin[:, None]
                query = rotate(q)
                torch.testing.assert_close(predicted[1], query)
                keys = F.linear(code, reference.aux.k.weight.detach()).view(2, n, 2, 16)
                keys = rms(keys, reference.aux.k_norm.variance_epsilon) * reference.aux.k_norm.weight.detach()
                keys = rotate(keys.transpose(1, 2)).repeat_interleave(2, dim=1)
                values = F.linear(code, reference.aux.v.weight.detach()).view(2, n, 2, 16)
                values = values.transpose(1, 2).repeat_interleave(2, dim=1)
                scores = (query @ keys.transpose(-1, -2)) / 4
                scores = scores.masked_fill(~strict, -torch.inf)
                weights = torch.where(strict.any(-1, keepdim=True), scores, 0).softmax(-1)
                weights = weights.masked_fill(~strict, 0)
                correction = F.linear((weights @ values).transpose(1, 2).reshape(2, n, 64),
                                      reference.aux.out.weight.detach())
                value = captured[4][0].detach() + correction
                use_mask = eligible & strict[:, 0, :-1].any(-1)
            else:
                value, use_mask = code, eligible
            extract = ce(reference.readout, value, labels.masked_fill(~use_mask, -100))
            lm = ce(reference.backbone.lm_head, hidden, labels)
            error = (predicted[0] - code.detach()).abs()
            align = torch.where(error < 1, .5 * error.square(), error - .5).mean(-1)[batch.valid].sum()
            torch.testing.assert_close(out['lm_sum'], lm)
            torch.testing.assert_close(out['extractor_sum'], extract)
            torch.testing.assert_close(out['align_sum'], align)
            loss = lm / eligible.sum() + extract / use_mask.sum()
            if reference.code_loss_weight:
                loss = loss + reference.code_loss_weight * align / batch.valid.sum()
            loss.backward()
            for (name, p), (_, q) in zip(actual.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_bf16_adam_resume_all_checkpoint_combinations(self):
        def step(m, optimizer, scheduler):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                loss = objective(m, m(context()))
            loss.backward()
            gradients = {n: p.grad.clone() for n, p in m.named_parameters()}
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            return loss.detach(), gradients

        for arm in BOTTLENECK_ARMS:
            m = model(arm, checkpoint=True)
            optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: 1 - .01 * i)
            for _ in range(2):
                step(m, optimizer, scheduler)
            saved = copy.deepcopy((m.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
            expected = step(m, optimizer, scheduler)
            for flags in itertools.product((False, True), repeat=3):
                with self.subTest(arm=arm, checkpoint=flags):
                    other = model(arm)
                    opt = torch.optim.AdamW(other.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.1)
                    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1 - .01 * i)
                    other.load_state_dict(saved[0])
                    opt.load_state_dict(copy.deepcopy(saved[1]))
                    sched.load_state_dict(saved[2])
                    other.checkpoint_layers, other.checkpoint_lm, other.checkpoint_aux = flags
                    actual = step(other, opt, sched)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(other.state_dict(), m.state_dict(), atol=1e-7, rtol=1e-5)
                    torch.testing.assert_close(opt.state_dict(), optimizer.state_dict(), atol=1e-7, rtol=1e-5)
                    self.assertEqual(sched.state_dict(), scheduler.state_dict())

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
        from tests.test_train import fixture_target_counts
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            target_count, consumer_count = fixture_target_counts(root, config)
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
                self.assertEqual(evaluation['eval_extractor_targets'], consumer_count if arm.startswith('Consumer') else target_count)
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
