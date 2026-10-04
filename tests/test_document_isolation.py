"""Local feasibility checks; production packing remains unchanged.

Defaults to CPU FP32. Set DOCUMENT_TEST_DEVICE=cuda:0 and
DOCUMENT_TEST_BF16=1 explicitly for an authorized local GPU check.
"""
import contextlib
import os
import unittest

import torch

from deep_kv import ARMS
from deep_kv.model import Context, DeepKV
from tests.test_deep_kv import config


def segmented_context(ids, eos=31):
    # Test-only prototype: an appended EOS belongs to the preceding document.
    ends = ids.eq(eos).long()
    segments = ends.cumsum(-1) - ends
    starts = torch.zeros_like(ids, dtype=torch.bool)
    starts[:, 0] = True
    starts[:, 1:] = ends[:, :-1].bool()
    offsets = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    positions = offsets - offsets.masked_fill(~starts, 0).cummax(-1).values
    return Context(ids, torch.ones_like(ids, dtype=torch.bool), positions, segments)


class DocumentIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.device = torch.device(os.environ.get('DOCUMENT_TEST_DEVICE', 'cpu'))
        cls.bf16 = os.environ.get('DOCUMENT_TEST_BF16') == '1'
        cls.tolerance = dict(atol=3e-3, rtol=1e-2) if cls.bf16 else dict(atol=2e-6, rtol=2e-5)

    def autocast(self):
        return (torch.autocast(self.device.type, dtype=torch.bfloat16)
                if self.bf16 else contextlib.nullcontext())

    def make_model(self, arm, checkpoint=False):
        model = DeepKV.from_scratch(config(), arm, consumer=2, deep_target=4,
            checkpoint_layers=checkpoint, checkpoint_lm=checkpoint,
            checkpoint_aux=checkpoint, lm_chunk=3).to(self.device)
        if model.aux is not None:
            # Exercise an active learned branch, not just its zero initialization.
            with torch.no_grad():
                model.aux.out.weight.fill_(0.03)
        return model

    def batch(self):
        return segmented_context(torch.tensor([[3, 4, 31, 8, 9, 10, 31]], device=self.device))

    def test_eos_ownership_positions_and_prediction_targets(self):
        batch = self.batch()
        self.assertEqual(batch.segments.tolist(), [[0, 0, 0, 1, 1, 1, 1]])
        self.assertEqual(batch.position_ids.tolist(), [[0, 1, 2, 0, 1, 2, 3]])
        # Predict both EOS tokens, but never predict B's first token from A's EOS.
        self.assertEqual(batch.targets().tolist(), [[True, True, False, True, True, True]])
        self.assertFalse(batch.allowed()[0, 0, 3:, :3].any())
        edge = segmented_context(torch.tensor([[31, 31, 3, 4, 5]], device=self.device))
        self.assertEqual(edge.segments.tolist(), [[0, 1, 2, 2, 2]])
        self.assertEqual(edge.position_ids.tolist(), [[0, 0, 0, 1, 2]])
        self.assertEqual(edge.targets().tolist(), [[False, False, True, True]])

    def test_all_arms_isolate_outputs_and_input_gradients(self):
        for arm in ARMS:
            for checkpoint in (False, True):
                with self.subTest(arm=arm, checkpoint=checkpoint):
                    model = self.make_model(arm, checkpoint).train()
                    batch = self.batch()
                    changed_ids = batch.input_ids.clone()
                    changed_ids[:, :2] = torch.tensor([15, 16], device=self.device)
                    changed = segmented_context(changed_ids)
                    with self.autocast():
                        before = model.hidden_states(batch)[0]
                        after = model.hidden_states(changed)[0]
                        torch.testing.assert_close(before[:, 3:], after[:, 3:], atol=0, rtol=0)
                        # Unique A token IDs let embedding gradients detect leakage.
                        before[:, 3:].float().square().sum().backward()
                    grad = model.backbone.model.embed_tokens.weight.grad
                    self.assertEqual(grad[[3, 4]].count_nonzero().item(), 0)
                    self.assertGreater(grad[[8, 9, 10]].abs().sum().item(), 0)

    def test_all_arms_packed_matches_separate_documents_and_backward(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = self.make_model(arm).train()
                batch = self.batch()
                with self.autocast():
                    together = model(batch)
                    parts = [model(segmented_context(batch.input_ids[:, a:b]))
                             for a, b in ((0, 3), (3, 7))]
                    # Includes LM and auxiliary sums and their eligible counts.
                    for key, value in together.items():
                        torch.testing.assert_close(value, parts[0][key] + parts[1][key], **self.tolerance)
                    loss = sum(value for key, value in together.items() if key.endswith('_sum'))
                    loss.backward()
                packed_gradients = {name: p.grad.detach().clone()
                                    for name, p in model.named_parameters() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                with self.autocast():
                    separate_loss = sum(value for part in parts for key, value in part.items()
                                        if key.endswith('_sum'))
                    separate_loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        torch.testing.assert_close(parameter.grad, packed_gradients.pop(name),
                                                   **self.tolerance, msg=name)
                self.assertFalse(packed_gradients, 'Different active parameters in packed/separate runs')

    def test_unsegmented_control_really_leaks_and_fast_path_refuses_segments(self):
        model = self.make_model('B').eval()
        batch = self.batch()
        changed = self.batch()
        changed.input_ids[:, :2] = torch.tensor([15, 16], device=self.device)
        batch.segments = changed.segments = None
        with torch.no_grad(), self.autocast():
            delta = (model.hidden_states(batch)[0][:, 3:] - model.hidden_states(changed)[0][:, 3:]).abs().max()
        self.assertGreater(delta.item(), 1e-4)
        model.causal_attention = True
        with self.assertRaisesRegex(ValueError, 'unsegmented'):
            model.hidden_states(self.batch())


if __name__ == '__main__':
    unittest.main()
