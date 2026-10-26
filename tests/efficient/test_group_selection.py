"""CPU correctness checks: algebra, group layout, causality and cached decoding."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
import copy
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from efficient_llm.group_selection import (GroupSelection, apply_mask, contributions,
                                         magnitude_mask, reconstruction_errors, subset_masks)
from efficient_llm.diagnostic_data import choose_documents
torch.set_num_threads(1)


class AlgebraTests(unittest.TestCase):
    def test_direct_and_gram_errors_and_projection_bias(self):
        torch.manual_seed(7)
        x, weight, bias = torch.randn(2, 3, 24), torch.randn(11, 24), torch.randn(11)
        c = contributions(x, weight, 4)
        torch.testing.assert_close(c.sum(-2) + bias, torch.nn.functional.linear(x, weight, bias))
        masks = subset_masks(4, 2)
        gram = reconstruction_errors(c, masks)
        for i, mask in enumerate(masks):
            direct = torch.nn.functional.linear(x - apply_mask(x, mask), weight).square().sum(-1)
            torch.testing.assert_close(gram[..., i], direct, rtol=2e-5, atol=2e-4)

    def test_known_cancellation_and_deterministic_ties(self):
        c = torch.tensor([[[10., 0], [-10., 0], [0, 1.], [0, 1.]]])
        masks = subset_masks(4, 2)
        errors = reconstruction_errors(c, masks)
        self.assertEqual(masks[errors.argmin()].tolist(), [False, False, True, True])
        self.assertEqual(errors.min().item(), 0)
        self.assertEqual(magnitude_mask(torch.zeros(1, 4), 2).tolist(), [[True, True, False, False]])

    def test_whole_gqa_group_mapping(self):
        x = torch.arange(24.).reshape(1, 1, 24)
        result = apply_mask(x, torch.tensor([True, False, True, False]))
        self.assertEqual(result.flatten().tolist(), list(range(6))+[0]*6+list(range(12,18))+[0]*6)

    def test_document_selection_reproducibility_and_disjoint_prefixes(self):
        data = [{"text": str(i)} for i in range(10)]
        def tokenizer(text, **kwargs): return {"input_ids": [int(text)+1]*6}
        kwargs = dict(count=3, length=5, seed=42, source="fixture", eod_id=99)
        rows, records = choose_documents(data, tokenizer, excluded=set(), **kwargs)
        self.assertEqual((rows, records), choose_documents(data, tokenizer, excluded=set(), **kwargs))
        seen = {r[k] for r in records for k in ("text_sha256", "tokens_sha256")}
        other, _ = choose_documents(data, tokenizer, excluded=seen, **kwargs)
        self.assertFalse({tuple(x) for x in rows} & {tuple(x) for x in other})


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        config = Qwen3Config(vocab_size=47, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4,
            head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
            attention_bias=True, tie_word_embeddings=False)
        self.model = Qwen3ForCausalLM._from_config(config, attn_implementation="sdpa").eval()
        self.ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13]])

    def test_keep_all_identity_and_hooks_removed_after_exception(self):
        with torch.inference_mode():
            expected = self.model(self.ids).logits
            for policy in ("dense", "raw_norm", "projected_norm", "joint"):
                with GroupSelection(self.model, keep=4, policy=policy):
                    torch.testing.assert_close(self.model(self.ids).logits, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with GroupSelection(self.model, keep=2, policy="joint"):
                raise RuntimeError("fixture")
        self.assertFalse(any(l.self_attn.o_proj._forward_pre_hooks for l in self.model.model.layers))

    def test_no_future_leakage_and_full_vs_cached_logits(self):
        masks = torch.tensor([[True, True, False, False]]*2)
        with torch.inference_mode():
            for policy in ("static", "raw_norm", "projected_norm", "joint"):
                with self.subTest(policy=policy), GroupSelection(self.model, keep=2, policy=policy, static_masks=masks):
                    full = self.model(self.ids, use_cache=False).logits
                    changed = self.ids.clone(); changed[:, 4:] = 21
                    torch.testing.assert_close(self.model(changed, use_cache=False).logits[:, :4], full[:, :4], rtol=2e-5, atol=2e-6)
                    cache, parts = None, []
                    for t in range(self.ids.shape[1]):
                        output = self.model(self.ids[:, t:t+1], past_key_values=cache, use_cache=True)
                        parts.append(output.logits); cache = output.past_key_values
                    torch.testing.assert_close(torch.cat(parts, 1), full, rtol=2e-4, atol=2e-5)
                    self.assertEqual(cache.layers[0].keys.shape[1], 4)
                    self.assertEqual(cache.get_seq_length(), self.ids.shape[1])

    def test_calibration_statistics_use_dense_model_and_static_coverage(self):
        with torch.inference_mode():
            expected = self.model(self.ids).logits
            with GroupSelection(self.model, keep=2, policy="dense", collect=True, sample_stride=2) as select:
                torch.testing.assert_close(self.model(self.ids).logits, expected, rtol=0, atol=0)
            self.assertEqual(set(select.stats), {0, 1})
            for row in select.stats.values():
                self.assertEqual(row["totals"][-1].item(), 3)
                self.assertEqual(len(row["candidate_error_sum"]), 6)
        with self.assertRaisesRegex(ValueError, "active-group count"):
            GroupSelection(self.model, keep=2, policy="static", static_masks=torch.ones(2,4,dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
