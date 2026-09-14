"""CPU checks for external diagnostic hooks; never run GPU-management code."""
import copy
import unittest
import numpy as np
import torch
from transformers import Qwen3Config
from ccm.model import MemoryLM
from scripts.memory_diagnostics import ContributionProbe, masks_for, check_pair, distribution


class DiagnosticTests(unittest.TestCase):
    @torch.no_grad()
    def test_normal_unchanged_off_equals_same_backbone_and_norms(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=8, max_position_embeddings=64, tie_word_embeddings=True,
                            attention_dropout=0., pad_token_id=0)
        config._attn_implementation = 'sdpa'
        for arm in ('contextual', 'isolated', 'grad'):
            model = MemoryLM(config, arm, table=torch.randn(3, 16), slots=3)
            model.reader.wv.weight.normal_(std=.05)
            model.set_phase('eval')
            base = MemoryLM(config)
            base.backbone.load_state_dict(model.backbone.state_dict())
            base.set_phase('eval')
            ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 0]])
            slots = torch.tensor([[-1, 0, 1, -1], [-1, 2, -1, -1]])
            inputs = dict(input_ids=ids, attention_mask=(ids != 0).long(),
                          position_ids=torch.arange(4).expand(2, -1), slots=slots,
                          targets=torch.tensor([[3, 4, 5, -100], [7, 8, -100, -100]]))
            masks = masks_for(dict(inputs, eligible=slots >= 0))
            original_state = copy.deepcopy(model.state_dict())
            original_slots = slots.clone()
            expected = model(**inputs, capture=True, return_logits=True)
            pre = model._states['r2_pre_memory'].clone()
            m = model.table[slots.clamp_min(0)].to(pre.dtype)
            contribution, _ = model.reader(pre, m)
            c = contribution[masks['hit']].float().norm(dim=-1)
            r = pre[masks['hit']].float().norm(dim=-1)
            probe = ContributionProbe(model.reader)
            try:
                probe.mask = masks['hit']
                got = model(**inputs, return_logits=True)
                self.assertTrue(torch.equal(expected['logits'], got['logits']))
                np.testing.assert_array_equal(probe.values, torch.stack((c, r, c/r), -1).numpy())
                probe.off = True
                disabled = model(**inputs, return_logits=True)
                baseline = base(**inputs, return_logits=True)
                self.assertTrue(torch.equal(disabled['logits'], baseline['logits']))
                self.assertFalse(torch.equal(got['logits'], disabled['logits']))
            finally:
                probe.close()
            self.assertTrue(torch.equal(slots, original_slots))
            self.assertTrue(all(torch.equal(v, original_state[k]) for k, v in model.state_dict().items()))
            self.assertTrue(torch.equal(expected['logits'], model(**inputs, return_logits=True)['logits']))

    def test_masks_and_pair_validation(self):
        b = dict(targets=torch.tensor([[1, 2, -100, 3]]), slots=torch.tensor([[0, -1, 1, -1]]),
                 eligible=torch.tensor([[True, True, True, False]]))
        m = masks_for(b)
        self.assertEqual({k: int(v.sum()) for k, v in m.items()},
                         dict(overall=3, hit=1, miss=2, eligible_miss=1))
        row = dict(doc_id='a', content_hash='b', segment_id=1,
                   overall=[6., 3], hit=[2., 1], miss=[4., 2], eligible_miss=[2., 1])
        self.assertEqual(check_pair(row, row, 0), 0)
        wrong = dict(row, doc_id='c')
        with self.assertRaises(ValueError):
            check_pair(row, wrong)
        wrong = dict(row, hit=[2., 2])
        with self.assertRaises(ValueError):
            check_pair(row, wrong)
        with self.assertRaises(ValueError):
            distribution(np.array([float('nan')]))


if __name__ == '__main__':
    unittest.main()
