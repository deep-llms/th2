import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from compositional.product_code import ProductCodeEmbed
from compositional.tied_head import TiedProductCodeHead
from eval.hashed_zh_diagnostics import (
    BucketMateIndex,
    analyze_logits,
    fixed_random_token_ids,
    gate_group_summary,
    load_language_counts,
    pairs_sharing_two_or_more,
    sample_random_pairs,
    summarize_prediction_values,
    validate_hashed_tied_model,
)


class HashedZhDiagnosticsTest(unittest.TestCase):
    def test_language_counts_use_per_language_normalization_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.npz"
            np.savez(
                path,
                counts_en=np.array([9, 0, 0, 1]),
                counts_zh=np.array([0, 3, 0, 1]),
            )
            languages, normalized, dominant = load_language_counts(path, 4)
        self.assertEqual(languages, ["en", "zh"])
        np.testing.assert_allclose(normalized.sum(axis=1), np.ones(2))
        np.testing.assert_array_equal(dominant, np.array([0, 1, -1, 1]))

    def test_bucket_mates_are_union_without_target(self):
        tail_ids = np.array([10, 11, 12, 13])
        codes = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ])
        index = BucketMateIndex(tail_ids, codes, num_buckets=2)
        np.testing.assert_array_equal(index.mate_rows(0), np.array([1, 2, 3]))
        np.testing.assert_array_equal(
            index.mate_token_ids(0), np.array([11, 12, 13])
        )
        expected = {(0, 1), (0, 2), (0, 3)}
        actual = set(map(tuple, pairs_sharing_two_or_more(codes, 2)))
        self.assertEqual(actual, expected)

    def test_random_baseline_is_deterministic_and_excludes_mates(self):
        tail_ids = np.arange(20, 30)
        codes = np.column_stack((np.arange(10) % 5, np.arange(10) // 2))
        index = BucketMateIndex(tail_ids, codes, num_buckets=5)
        mates = set(index.mate_token_ids(0).tolist())
        first = fixed_random_token_ids(index, 0, 3, seed=7)
        second = fixed_random_token_ids(index, 0, 3, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(set(first.tolist())), 3)
        self.assertFalse(set(first.tolist()) & mates)
        self.assertNotIn(int(tail_ids[0]), first.tolist())

    def test_random_cosine_pairs_are_language_matched_nonmates(self):
        codes = np.column_stack((np.arange(30) % 5, np.arange(30) // 5))
        dominant = np.array([1] * 10 + [0] * 20)
        pairs = sample_random_pairs(
            codes, dominant, zh_index=1, sample_size=8, seed=11
        )
        for category, values in pairs.items():
            self.assertEqual(values.shape, (8, 2))
            for left, right in values:
                self.assertFalse(np.any(codes[left] == codes[right]))
                labels = (dominant[left], dominant[right])
                if category == "zh_zh":
                    self.assertEqual(labels, (1, 1))
                elif category == "zh_other":
                    self.assertEqual(set(labels), {0, 1})
                else:
                    self.assertEqual(labels, (0, 0))

    def test_checkpoint_validation_requires_exact_hashed_tying(self):
        embed = ProductCodeEmbed(
            vocab_size=20,
            embed_dim=8,
            head_size=4,
            num_hashes=2,
            num_buckets=5,
            importance=torch.arange(20, dtype=torch.float64),
            assignment="hashed",
            seed=3,
        )
        model = SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=SimpleNamespace(embed=embed)
            ),
            lm_head=TiedProductCodeHead(embed),
        )
        config = {
            "arm": "product_code",
            "product_code_assignment": "hashed",
            "product_code_seed": 3,
            "tie_output": True,
        }
        self.assertIs(validate_hashed_tied_model(model, config), embed)

        with self.assertRaisesRegex(ValueError, "exactly tied"):
            validate_hashed_tied_model(model, {**config, "tie_output": False})

        embed.codes[0, 0] = (embed.codes[0, 0] + 1) % embed.num_buckets
        with self.assertRaisesRegex(ValueError, "deterministic hashed"):
            validate_hashed_tied_model(model, config)

    def test_logit_diagnostics_match_explicit_softmax(self):
        logits = torch.tensor([
            [3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        ])
        targets = np.array([0, 3])
        mates = [np.array([1, 2]), np.array([4])]
        randoms = [np.array([4, 5]), np.array([0])]
        values = analyze_logits(
            logits, targets, mates, randoms, batch_size=1
        )
        probabilities = logits.softmax(dim=-1).numpy()
        expected_mates = np.array([
            probabilities[0, [1, 2]].sum(), probabilities[1, 4]
        ])
        expected_random = np.array([
            probabilities[0, [4, 5]].sum(), probabilities[1, 0]
        ])
        np.testing.assert_allclose(values["p_mates"], expected_mates, rtol=1e-6)
        np.testing.assert_allclose(values["p_random"], expected_random, rtol=1e-6)
        np.testing.assert_array_equal(values["target_rank"], np.array([1, 3]))
        np.testing.assert_array_equal(
            values["argmax_is_mate"], np.array([False, False])
        )
        summary = summarize_prediction_values(values)
        self.assertEqual(summary["num_tail_positions"], 2)
        self.assertAlmostEqual(
            summary["mean_mates_over_random"],
            float(np.mean(expected_mates / expected_random)),
            places=5,
        )

    def test_gate_summary_reports_token_extremes(self):
        summary = gate_group_summary(np.array([
            [1.0, 1.0],
            [0.05, 1.0],
            [1.0, 3.5],
        ]))
        self.assertEqual(summary["num_tokens"], 3)
        self.assertAlmostEqual(summary["fraction_any_below_0.1"], 1 / 3)
        self.assertAlmostEqual(summary["fraction_any_above_3"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
